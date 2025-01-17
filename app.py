import streamlit as st
import requests
from dotenv import load_dotenv
import os
import pandas as pd
import plotly.express as px
from datetime import datetime

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

# Load environment variables
load_dotenv()

# Constants
BASE_URL = "https://api.rawg.io/api"
API_KEY = os.getenv("RAWG_API_KEY")

# Cache API calls to improve performance
@st.cache_data(ttl=3600)
def search_game(game_name):
    """Search for a game by name"""
    params = {
        'key': API_KEY,
        'search': game_name,
        'page_size': 5
    }
    response = requests.get(f"{BASE_URL}/games", params=params)
    return response.json()

@st.cache_data(ttl=3600)
def get_game_details(game_id):
    """Get detailed information about a specific game"""
    params = {
        'key': API_KEY
    }
    response = requests.get(f"{BASE_URL}/games/{game_id}", params=params)
    return response.json()

@st.cache_data(ttl=3600)
def get_genres():
    """Get list of available genres"""
    params = {
        'key': API_KEY
    }
    response = requests.get(f"{BASE_URL}/genres", params=params)
    return response.json()['results']

def calculate_game_score(game, preferences):
    """Enhanced recommendation scoring system"""
    score = 0
    max_score = 100
    
    # 1. Genre Matching (0-25 points)
    if preferences.get('genres'):
        game_genre_names = {genre['name'].lower() for genre in game.get('genres', [])}
        pref_genre_names = {genre.lower() for genre in preferences['genres']}
        
        if game_genre_names and pref_genre_names:
            genre_match_ratio = len(game_genre_names.intersection(pref_genre_names)) / len(pref_genre_names)
            genre_score = genre_match_ratio * 25
            score += genre_score
    
    # 2. Rating-based scoring (0-20 points)
    rating = game.get('rating', 0)
    if rating:
        # Weighted more heavily if it exceeds the threshold
        if rating >= preferences.get('rating_threshold', 0):
            rating_score = (float(rating) / 5) * 20
        else:
            rating_score = (float(rating) / 5) * 10
        score += rating_score
    
    # 3. Similarity to Favorite Games (0-25 points)
    similarity_score = 0
    for fav_game in preferences['favorite_games']:
        # Genre similarity
        fav_genres = {genre['name'].lower() for genre in fav_game.get('genres', [])}
        current_genres = {genre['name'].lower() for genre in game.get('genres', [])}
        genre_similarity = len(fav_genres.intersection(current_genres)) / max(len(fav_genres), 1)
        
        # Tags similarity
        fav_tags = {tag['name'].lower() for tag in fav_game.get('tags', [])}
        current_tags = {tag['name'].lower() for tag in game.get('tags', [])}
        tag_similarity = len(fav_tags.intersection(current_tags)) / max(len(fav_tags), 1)
        
        # Combine similarities
        game_similarity = (genre_similarity * 0.6 + tag_similarity * 0.4) * 25
        similarity_score = max(similarity_score, game_similarity)
    
    score += similarity_score
    
    # 4. Relevance Factors (0-20 points)
    relevance_score = 0
    
    # Metacritic score impact
    metacritic = game.get('metacritic')
    if metacritic:
        metacritic_score = (float(metacritic) / 100) * 10
        relevance_score += metacritic_score
    
    # Release date relevance (newer games score slightly higher)
    if game.get('released'):
        try:
            release_date = datetime.strptime(game['released'], '%Y-%m-%d')
            years_old = (datetime.now() - release_date).days / 365
            if years_old <= 1:  # Released within last year
                relevance_score += 10
            elif years_old <= 3:  # Released within last 3 years
                relevance_score += 7
            elif years_old <= 5:  # Released within last 5 years
                relevance_score += 5
        except:
            pass
    
    score += relevance_score
    
    # 5. Player Count Preference Adjustment
    player_preference = preferences.get('filters', {}).get('player_count')
    if player_preference != 'Any':
        tag_names = {tag['name'].lower() for tag in game.get('tags', [])}
        
        if player_preference == 'Single-player only':
            if 'singleplayer' not in tag_names:
                score *= 0.4
            elif 'multiplayer' in tag_names:
                score *= 0.8
        elif player_preference == 'Multiplayer only':
            if 'multiplayer' not in tag_names:
                score *= 0.4
            elif 'singleplayer' in tag_names and 'multiplayer' not in tag_names:
                score *= 0.8
    
    # 6. Apply User Weights
    weights = preferences.get('weights', {})
    
    # Gameplay weight affects rating score
    gameplay_weight = weights.get('gameplay', 3)
    score = score * (0.8 + (gameplay_weight / 10))
    
    # Graphics weight affects metacritic and release date relevance
    graphics_weight = weights.get('graphics', 3)
    if metacritic or game.get('released'):
        score = score * (0.8 + (graphics_weight / 10))
    
    # Story weight affects genre and tag matching
    story_weight = weights.get('story', 3)
    if any(tag['name'].lower() in ['story-rich', 'narrative'] for tag in game.get('tags', [])):
        score = score * (0.8 + (story_weight / 10))
    
    return min(score, max_score)

def get_recommendations(preferences):
    """Get game recommendations based on preferences"""
    params = {
        'key': API_KEY,
        'page_size': 40,
        'ordering': '-rating'
    }
    
    try:
        response = requests.get(f"{BASE_URL}/games", params=params)
        games = response.json()['results']
        
        # Score and filter games
        scored_games = []
        for game in games:
            # Skip games that are in favorite games
            if not any(fg['id'] == game['id'] for fg in preferences['favorite_games']):
                try:
                    score = calculate_game_score(game, preferences)
                    scored_games.append((game, score))
                except Exception as e:
                    st.write(f"Error scoring game {game.get('name', 'Unknown')}: {str(e)}")
                    continue
        
        # Sort by score
        scored_games.sort(key=lambda x: x[1], reverse=True)
        return scored_games[:10]
    
    except Exception as e:
        st.write(f"Error getting recommendations: {str(e)}")
        return []

def compare_games(game1, game2):
    """Compare two games side by side"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(game1['name'])
        if game1.get('background_image'):
            st.image(game1['background_image'])
        
        # Game details
        st.write("📊 **Rating:**", f"{game1.get('rating', 'N/A')}/5")
        st.write("📅 **Released:**", game1.get('released', 'N/A'))
        
        if game1.get('genres'):
            st.write("🎯 **Genres:**", ', '.join(g['name'] for g in game1['genres']))
        
        if game1.get('platforms'):
            platforms = ', '.join(p['platform']['name'] for p in game1['platforms'])
            st.write("🎮 **Platforms:**", platforms)
        
        # Get detailed info
        details1 = get_game_details(game1['id'])
        if details1.get('metacritic'):
            st.write("📈 **Metacritic:**", details1['metacritic'])
        
        if details1.get('playtime'):
            st.write("⏱️ **Average playtime:**", f"{details1['playtime']} hours")
            
    with col2:
        st.subheader(game2['name'])
        if game2.get('background_image'):
            st.image(game2['background_image'])
        
        # Game details
        st.write("📊 **Rating:**", f"{game2.get('rating', 'N/A')}/5")
        st.write("📅 **Released:**", game2.get('released', 'N/A'))
        
        if game2.get('genres'):
            st.write("🎯 **Genres:**", ', '.join(g['name'] for g in game2['genres']))
        
        if game2.get('platforms'):
            platforms = ', '.join(p['platform']['name'] for p in game2['platforms'])
            st.write("🎮 **Platforms:**", platforms)
        
        # Get detailed info
        details2 = get_game_details(game2['id'])
        if details2.get('metacritic'):
            st.write("📈 **Metacritic:**", details2['metacritic'])
        
        if details2.get('playtime'):
            st.write("⏱️ **Average playtime:**", f"{details2['playtime']} hours")

    # Show common features
    st.subheader("Comparison")
    
    # Compare genres
    genres1 = {g['name'].lower() for g in game1.get('genres', [])}
    genres2 = {g['name'].lower() for g in game2.get('genres', [])}
    common_genres = genres1.intersection(genres2)
    
    if common_genres:
        st.write("🎯 **Shared Genres:**", ', '.join(common_genres))
    
    # Compare tags
    tags1 = {t['name'].lower() for t in game1.get('tags', [])}
    tags2 = {t['name'].lower() for t in game2.get('tags', [])}
    common_tags = tags1.intersection(tags2)
    
    if common_tags:
        st.write("🏷️ **Shared Features:**", ', '.join(common_tags))

def sort_recommendations(recommendations, sort_option):
    """Sort recommendations based on selected option"""
    if sort_option == "Match Score":
        return sorted(recommendations, key=lambda x: x[1], reverse=True)
    elif sort_option == "Release Date":
        return sorted(recommendations, key=lambda x: x[0].get('released', ''), reverse=True)
    elif sort_option == "Rating":
        return sorted(recommendations, key=lambda x: x[0].get('rating', 0), reverse=True)
    elif sort_option == "Name":
        return sorted(recommendations, key=lambda x: x[0].get('name', ''))
    return recommendations

def filter_recommendations(recommendations, filter_options):
    """Filter recommendations based on selected options"""
    filtered = recommendations.copy()
    
    for option in filter_options:
        if option == "Recent Games Only":
            current_year = datetime.now().year
            filtered = [(game, score) for game, score in filtered 
                       if game.get('released') and int(game['released'][:4]) >= current_year - 2]
        
        elif option == "Highly Rated":
            filtered = [(game, score) for game, score in filtered 
                       if game.get('rating', 0) >= 4]
        
        elif option == "Popular":
            filtered = [(game, score) for game, score in filtered 
                       if game.get('ratings_count', 0) > 100]
        
        elif option == "Classic Games":
            current_year = datetime.now().year
            filtered = [(game, score) for game, score in filtered 
                       if game.get('released') and 
                       int(game['released'][:4]) <= current_year - 5 and 
                       game.get('rating', 0) >= 4]
    
    return filtered


def display_game_card(game, score=None):
    """Display a game card with details"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if game.get('background_image'):
            st.image(game['background_image'], use_container_width=True)
    
    with col2:
        st.subheader(game['name'])
        if score is not None:
            st.write(f"Match Score: {score:.1f}%")
        st.write(f"Rating: {game.get('rating', 'N/A')}/5")
        st.write(f"Released: {game.get('released', 'N/A')}")
        
        if game.get('genres'):
            genres = ', '.join(genre['name'] for genre in game['genres'])
            st.write(f"Genres: {genres}")
        
        if game.get('platforms'):
            platforms = ', '.join(p['platform']['name'] for p in game['platforms'])
            st.write(f"Platforms: {platforms}")
        
        if st.button(f"More Details 🔍", key=f"details_{game['id']}"):
            details = get_game_details(game['id'])
            st.write("---")
            st.write("Description:")
            st.write(details.get('description_raw', 'No description available'))
            
            if details.get('metacritic'):
                st.write(f"Metacritic Score: {details['metacritic']}")
            
            if details.get('website'):
                st.write(f"Website: {details['website']}")

def main():
    # Page config
    st.set_page_config(page_title="Game Recommender", page_icon="🎮", layout="wide")
    
    # Title and description
    st.title("🎮 Game Recommendation Engine")
    st.markdown("Find your next favorite game based on your preferences!")
    
    # Sidebar
    with st.sidebar:
        st.header("Your Gaming Preferences")
        
        # Favorite games input with autocomplete
        st.subheader("Favorite Games")
        favorite_games = []
        for i in range(3):
            game_name = st.text_input(f"Favorite Game {i+1}", key=f"game_{i}")
            if game_name:
                results = search_game(game_name)
                if results['results']:
                    options = [g['name'] for g in results['results']]
                    selected = st.selectbox(f"Select exact game {i+1}", options, key=f"select_{i}")
                    game = next(g for g in results['results'] if g['name'] == selected)
                    favorite_games.append(game)
        
        # Genre selection
        st.subheader("Preferred Genres")
        genres = [genre['name'] for genre in get_genres()]
        selected_genres = st.multiselect("Select genres", genres)
        
        # Rating threshold
        rating_threshold = st.slider("Minimum Rating", 0.0, 5.0, 4.0, 0.1)
        
        # Advanced filters
        st.subheader("Advanced Filters")
        player_count = st.radio("Player Count", ['Any', 'Single-player only', 'Multiplayer only'])
        
        # Importance weights
        st.subheader("Factor Importance")
        gameplay_weight = st.slider("Gameplay Importance", 1, 5, 3)
        graphics_weight = st.slider("Graphics Importance", 1, 5, 3)
        story_weight = st.slider("Story Importance", 1, 5, 3)

    if favorite_games:
        st.subheader("Your Favorite Games")
        for game in favorite_games:
            display_game_card(game)
                
    # Main content
    if st.button("Get Recommendations", type="primary"):
        if not favorite_games:
            st.warning("Please enter at least one favorite game!")
            return
        
        preferences = {
            'favorite_games': favorite_games,
            'genres': selected_genres,
            'rating_threshold': rating_threshold,
            'filters': {'player_count': player_count},
            'weights': {
                'gameplay': gameplay_weight,
                'graphics': graphics_weight,
                'story': story_weight
            }
        }
        
        with st.spinner("Finding the best games for you..."):
            st.session_state.recommendations = get_recommendations(preferences)

    # Display recommendations
    if st.session_state.recommendations:
        st.subheader("Recommended Games")
        
        # Add sorting options
        col1, col2 = st.columns([2, 3])
        with col1:
            sort_option = st.selectbox(
                "Sort by:",
                ["Match Score", "Release Date", "Rating", "Name"]
            )
        
        # Add filtering options
        with col2:
            filter_options = st.multiselect(
                "Filter by:",
                ["Recent Games Only", "Highly Rated", "Popular", "Classic Games"]
            )
        
        # Apply sorting and filtering
        filtered_recommendations = filter_recommendations(st.session_state.recommendations, filter_options)
        sorted_recommendations = sort_recommendations(filtered_recommendations, sort_option)
        
        # Show number of results
        st.write(f"Showing {len(sorted_recommendations)} recommendations")
        
        if not sorted_recommendations:
            st.info("No games match the selected filters. Try adjusting your filter options.")
        else:
            # Analytics Section
            st.write("---")
            st.header("📈 Analytics Dashboard")
            
            # Convert recommendations to DataFrame for analytics
            games_data = pd.DataFrame([
                {
                    'name': game['name'],
                    'rating': game.get('rating', 0),
                    'released': game.get('released'),
                    'match_score': score,
                    'genres': ', '.join(g['name'] for g in game.get('genres', [])),
                    'metacritic': game.get('metacritic'),
                    'playtime': game.get('playtime', 0)
                }
                for game, score in sorted_recommendations
            ])
            
            # Create analytics tabs
            analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs([
                "General Statistics", 
                "Visualizations",
                "Platform Analysis"
            ])
            
            with analytics_tab1:
                # Basic statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_rating = games_data['rating'].mean()
                    st.metric("Average Rating", f"{avg_rating:.1f}/5")
                with col2:
                    avg_score = games_data['match_score'].mean()
                    st.metric("Average Match Score", f"{avg_score:.1f}%")
                with col3:
                    if 'released' in games_data.columns:
                        games_data['released'] = pd.to_datetime(games_data['released'])
                        years_range = f"{games_data['released'].dt.year.min()} - {games_data['released'].dt.year.max()}"
                        st.metric("Release Years Range", years_range)
            
            with analytics_tab2:
                # Ratings vs Match Scores scatter plot
                fig1 = px.scatter(games_data, 
                                x='rating', 
                                y='match_score',
                                text='name',
                                title='Game Ratings vs Match Scores',
                                labels={'rating': 'Game Rating', 
                                       'match_score': 'Match Score',
                                       'name': 'Game'})
                fig1.update_traces(textposition='top center')
                st.plotly_chart(fig1, use_container_width=True)
                
                # Genre distribution pie chart
                genres_list = [genre.strip() for genres in games_data['genres'].str.split(',') 
                             for genre in genres]
                genre_counts = pd.Series(genres_list).value_counts()
                fig2 = px.pie(values=genre_counts.values, 
                            names=genre_counts.index,
                            title='Genre Distribution in Recommendations')
                st.plotly_chart(fig2, use_container_width=True)
            
            with analytics_tab3:
                # Platform analysis
                platform_data = []
                for game, _ in sorted_recommendations:
                    if game.get('platforms'):
                        for platform in game['platforms']:
                            platform_data.append({
                                'game': game['name'],
                                'platform': platform['platform']['name']
                            })
                
                if platform_data:
                    df = pd.DataFrame(platform_data)
                    platform_counts = df['platform'].value_counts()
                    fig3 = px.bar(x=platform_counts.index, 
                                y=platform_counts.values,
                                title='Games Available by Platform',
                                labels={'x': 'Platform', 'y': 'Number of Games'})
                    st.plotly_chart(fig3, use_container_width=True)
            
            st.write("---")
            
            # Game Comparison Section
            st.subheader("Compare Games")
            col1, col2 = st.columns(2)
            
            # Get list of game names
            game_names = [game['name'] for game, _ in sorted_recommendations]
            
            with col1:
                game1_index = st.selectbox("Select first game", 
                                         range(len(game_names)), 
                                         format_func=lambda x: game_names[x],
                                         key="game1")
            
            with col2:
                game2_index = st.selectbox("Select second game", 
                                         range(len(game_names)), 
                                         format_func=lambda x: game_names[x],
                                         key="game2")
            
            if st.button("Compare Selected Games"):
                if game1_index != game2_index:
                    game1 = sorted_recommendations[game1_index][0]
                    game2 = sorted_recommendations[game2_index][0]
                    compare_games(game1, game2)
                else:
                    st.warning("Please select different games to compare")
            
            st.write("---")
            
            # Display all recommendations
            st.subheader("All Recommendations")
            for game, score in sorted_recommendations:
                st.write("---")
                display_game_card(game, score)

if __name__ == "__main__":
    main()
