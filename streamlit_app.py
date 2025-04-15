import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import os
import io
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Create NLTK data directory if it doesn't exist
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Configure Streamlit
st.set_page_config(
    page_title="EMA Bénin 2025 Text Analyzer",
    page_icon="🇧🇯",
    layout="wide"
)

# Download NLTK resources safely
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK resources: {e}")
        return False

# Try to download resources
resource_status = download_nltk_resources()

# Title and description with EMA Bénin 2025
st.title("📊 EMA Bénin 2025 - Text Analysis & Visualization Tool")
st.markdown("""
This app allows you to analyze and visualize text data. Upload a text file or input text directly,
customize parameters, and explore the results through word clouds and frequency charts.
""")

# Input section
st.sidebar.header("Input Options")
input_option = st.sidebar.radio("Choose input method:", ["Enter Text", "Upload Text File", "Use Sample Text"])

text_input = ""

if input_option == "Enter Text":
    text_input = st.sidebar.text_area(
        "Enter your text here:", 
        height=300, 
        placeholder="Paste or type your text here for analysis. For best results, include at least a few paragraphs of text in French or your chosen language."
    )
elif input_option == "Upload Text File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload a text file", 
        type=['txt'],
        help="Upload a plain text (.txt) file. Maximum size: 200MB"
    )
    # Add a placeholder message below the uploader
    if not uploaded_file:
        st.sidebar.info("📄 Upload a text file to analyze its content. Common formats like articles, speeches, or reports work best.")
    
    if uploaded_file is not None:
        try:
            text_input = uploaded_file.getvalue().decode("utf-8")
            st.sidebar.success("File uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
else:
    # Sample text option
    sample_option = st.sidebar.selectbox("Choose a sample text:", 
                                       ["Bénin Presidential Speech (Boni Yayi 2006)", "USA Presidential Speech (Donald Trump 2025)", "Technical Document"])
    
    if sample_option == "Bénin Presidential Speech (Boni Yayi 2006)":
        with open("samples/discour_boni_yayi_2006.txt", "r") as f:
            text_input = f.read()
        selected_language = "french"
        # text_input = """
        # En cette circonstance solennelle, que je vis avec beaucoup d'émotion et surtout d'espoir et d'espérance, 
        # je voudrais rendre grâce à Dieu, le Tout-Puissant, pour son infinie bonté et sa miséricorde à l'endroit du Bénin, 
        # notre chère patrie. J'exprime toute ma gratitude, et ce avec beaucoup de respect et de considération au vaillant 
        # peuple béninois qui a bien voulu me renouveler sa confiance, pour continuer de conduire ensemble notre nation sur 
        # le chemin de l'unité, de la paix, du progrès, de la solidarité et de la prospérité.
        
        # Excellences Madame, Messieurs les Chefs d'Etat, Distingués Représentants des Chefs d'Etat,
        # Au nom du peuple béninois, je vous remercie du fond du cœur d'être venus rehausser de votre présence, 
        # la cérémonie de ce jour.
        # """
    elif sample_option == "USA Presidential Speech (Donald Trump 2025)":
        with open("samples/discour_trump_2025.txt", "r") as f:
            text_input = f.read()
        selected_language = "english"
    else:
        text_input = """
        La complexité de l'algorithme est O(n log n) dans le cas moyen mais se dégrade à O(n²) dans le pire des cas.
        Le traitement des données nécessite une gestion efficace de la mémoire et des techniques d'optimisation.
        L'architecture du système comprend trois composants principaux: l'entrée de données, le moteur de traitement et la visualisation des résultats.
        Les utilisateurs peuvent configurer les paramètres via l'interface en ligne de commande ou l'API REST.
        Les détails d'implémentation sont documentés dans le document de spécifications techniques.
        """
    
    st.sidebar.success(f"Sample text loaded: {sample_option}")

# Processing options
st.sidebar.header("Processing Options")

# Language selection for stopwords
available_languages = ['english', 'french', 'spanish', 'german']
language_options = {
    'english': 'English', 
    'french': 'French', 
    'spanish': 'Spanish', 
    'german': 'German'
}

# Only show languages if NLTK resources loaded successfully
if resource_status:
    try:
        # Verify available stopword languages
        available_languages = [lang for lang in available_languages if lang in stopwords.fileids()]
        language_options = {lang: language_options[lang] for lang in available_languages}
    except:
        # Fallback if there's an error
        available_languages = ['english']
        language_options = {'english': 'English'}

# Default to French for Bénin
default_language_index = available_languages.index('french') if 'french' in available_languages else 0

selected_language = st.sidebar.selectbox("Select language for stopwords:", 
                                        list(language_options.keys()),
                                        index=default_language_index,
                                        format_func=lambda x: language_options[x])

# Remove stopwords option
remove_stopwords = st.sidebar.checkbox("Remove stopwords", value=True, help="Filter out common words like 'le', 'la', 'et', etc.")

# Additional stopwords input with placeholder
additional_stopwords = st.sidebar.text_input(
    "Additional stopwords (comma separated):", 
    "", 
    placeholder="e.g., bénin, aujourd'hui, monsieur, madame"
)
additional_stopwords_list = [word.strip() for word in additional_stopwords.split(',') if word.strip()] if additional_stopwords else []

# Set minimum word frequency
min_word_freq = st.sidebar.slider("Minimum word frequency:", 1, 50, 2, help="Only show words that appear at least this many times")

# Set maximum number of words to display
max_words_to_display = st.sidebar.slider("Maximum words to display in chart:", 10, 500, 100)

# Word cloud customization
st.sidebar.header("Word Cloud Options")
wc_width = st.sidebar.slider("Width:", 400, 1200, 800)
wc_height = st.sidebar.slider("Height:", 200, 800, 400)
wc_background = st.sidebar.color_picker("Background color:", "#FFFFFF")

# Custom color map - Bénin flag colors inspired
color_options = {
    "Greens": "Green (Bénin Flag)",
    "YlOrRd": "Yellow-Red (Bénin Flag)",
    "default": "Default",
    "viridis": "Viridis (Purple-Green-Yellow)", 
    "plasma": "Plasma (Purple-Red-Yellow)",
    "inferno": "Inferno (Black-Red-Yellow)",
    "magma": "Magma (Black-Red-White)",
    "blues": "Blues",
    "reds": "Reds"
}
selected_colormap = st.sidebar.selectbox("Color scheme:", 
                                      list(color_options.keys()),
                                      index=0,  # Default to Greens for Bénin
                                      format_func=lambda x: color_options[x])

# Function to process text safely
def process_text(text, remove_stops=True, language='english', additional_stops=None):
    if not text or not text.strip():
        return None, None
    
    try:
        # Lowercase and replace punctuation with spaces
        text = text.lower()
        for char in [',', '.', "'", '"', ';', ':', '!', '?', '(', ')', '[', ']', '{', '}', '\n', '\t', '’']:
            text = text.replace(char, ' ')
        
        # Split into words
        words = [word for word in text.split() if word.strip()]
        
        # Remove stopwords if requested
        if remove_stops and resource_status:
            try:
                stop_words = set(stopwords.words(language))
                if additional_stops:
                    stop_words.update(additional_stops)
                words = [word for word in words if word not in stop_words]
            except Exception as e:
                st.warning(f"Error loading stopwords: {e}. Proceeding without stopword removal.")
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Create dataframe
        df = pd.DataFrame(word_counts.items(), columns=['word', 'count'])
        df = df.sort_values(by="count", ascending=False)
        
        return word_counts, df
    
    except Exception as e:
        st.error(f"Error processing text: {e}")
        return None, None

# Main content area
if text_input:
    # Process the text
    word_counts, df = process_text(
        text_input, 
        remove_stops=remove_stopwords,
        language=selected_language,
        additional_stops=additional_stopwords_list
    )
    
    if word_counts and len(word_counts) > 0:
        # Filter by minimum frequency
        filtered_counts = {word: count for word, count in word_counts.items() if count >= min_word_freq}
        
        if filtered_counts:
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Word Cloud", "Bar Chart", "Data Table"])
            
            with tab1:
                st.header("Word Cloud Visualization")
                
                try:
                    # Generate word cloud
                    from wordcloud import WordCloud
                    import matplotlib.pyplot as plt
                    import matplotlib.patheffects as path_effects
                    
                    # --- Generate the WordCloud ---
                    wordcloud = WordCloud(
                        width=wc_width,
                        height=wc_height,
                        background_color=wc_background,  # try "white", "#f5f5f5", or "black" for contrast
                        max_words=max_words_to_display,
                        colormap=selected_colormap if selected_colormap != "default" else "viridis",  # use a smooth colormap
                        prefer_horizontal=0.9,
                        contour_color='steelblue',  # adds an outline
                        contour_width=1.5,
                        random_state=42,  # reproducibility
                        # font_path='arial.ttf'  # optional: use custom font
                    ).generate_from_frequencies(filtered_counts)
                    
                    # --- Display the WordCloud ---
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    
                    # Stylish Title
                    title = ax.set_title(
                        "✨ EMA Bénin 2025 – Word Cloud Analysis ✨",
                        fontsize=12,
                        color="#2c3e50",
                        pad=20,
                        weight='bold'
                    )
                    
                    # Add subtle shadow to title
                    title.set_path_effects([
                        path_effects.Stroke(linewidth=1.5, foreground='white'),
                        path_effects.Normal()
                    ])
                    
                    # Set tight layout and soft background
                    fig.patch.set_facecolor('#f0f0f0')
                    fig.tight_layout(pad=2)
                    
                    st.pyplot(fig)

                    
                    # Download option

                    # Save word cloud to buffer
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format='png')
                    buffer.seek(0)
                    
                    # Download option
                    st.download_button(
                        label="Download Word Cloud Image",
                        data=buffer,
                        file_name="ema_benin_2025_wordcloud.png",
                        mime="image/png"
                    )

                except Exception as e:
                    st.error(f"Error generating word cloud: {e}")
                
            with tab2:
                st.header("Word Frequency Bar Chart")
                
                # Create dataframe from filtered counts
                filtered_df = pd.DataFrame(filtered_counts.items(), columns=['word', 'count'])
                filtered_df = filtered_df.sort_values(by="count", ascending=False).head(max_words_to_display)
                
                try:
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, max(5, len(filtered_df) * 0.25)))
                    bars = ax.barh(filtered_df['word'], filtered_df['count'], 
                                   color=plt.cm.get_cmap(selected_colormap if selected_colormap != "default" else "viridis")(
                                       range(len(filtered_df)))
                                  )
                    ax.set_xlabel('Frequency')
                    ax.set_ylabel('Words')
                    ax.set_title('EMA Bénin 2025 - Word Frequency Distribution')
                    
                    # Add labels to bars
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                                f'{width}', ha='left', va='center')
                    
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating bar chart: {e}")
                
                # Download option for chart data
                try:
                    csv_data = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download Word Frequency Data (CSV)",
                        data=csv_data,
                        file_name="ema_benin_2025_word_frequencies.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error preparing download: {e}")
            
            with tab3:
                st.header("Word Frequency Data")
                st.dataframe(
                    filtered_df.head(max_words_to_display), 
                    use_container_width=True
                )
                
                # Show statistics
                st.subheader("Text Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Words", len(text_input.split()))
                
                with col2:
                    st.metric("Unique Words", len(word_counts))
                
                with col3:
                    st.metric("Words After Filtering", len(filtered_counts))
        else:
            st.warning("No words meet the minimum frequency threshold. Try lowering the minimum frequency.")
    else:
        st.warning("No words found after processing. Check your input and processing options.")
else:
    # Add a placeholder message when no text is provided
    st.info("Please enter some text, upload a file, or select a sample text to begin analysis.")
    
    # Display a placeholder visualization
    st.markdown("### Preview of Word Cloud Visualization")
    st.image("https://via.placeholder.com/800x400?text=Word+Cloud+Preview", 
             caption="A sample word cloud will appear here after text analysis")

# Footer
st.markdown("---")
st.markdown("🇧🇯 EMA Bénin 2025 - Text Analysis & Visualization Tool", help="A tool for analyzing and visualizing text data")
