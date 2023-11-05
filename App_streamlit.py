from st_pages import Page, show_pages, add_page_title

# Optional -- adds the title and icon to the current page
add_page_title("La sécurité routière, même à vélo !")

# Specify what pages should be shown in the sidebar, and what their titles and icons
# should be
show_pages(
    [
        Page("Page1_stream.py", "Home", "🏠"),
        Page("Page2_stream.py", "Page 2", ":books:"),
        Page("Page3_stream.py", "Test")
    ]
)