from st_pages import Page, show_pages, add_page_title
import streamlit as st

# Optional -- adds the title and icon to the current page
add_page_title("Restez vigilant, même à vélo !")

# Liste des pages
pages = [
    Page("Page1_stream.py", "Quelques statistiques générales", "🚴‍♂️"),
    Page("Page2_stream.py", "Qu'en est-il dans votre région ?", "🗺️"),
    Page("Page3_stream.py", "Vos statistiques personnalisées", "📊")
]

# Affichez les pages dans la barre latérale
show_pages(pages[:3])  # Affichez les trois premières pages