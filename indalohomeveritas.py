import streamlit as st
import streamlit.components.v1 as components

def indalohome():

    def add_vertical_space(num_lines: int = 1) -> None:
        """
        Add vertical space to your Streamlit app.

        Args:
            num_lines (int, optional): Height of the vertical space (given in number of lines). Defaults to 1.
        """
        for _ in range(num_lines):
            st.write("")  # This is just a way to do a line break!

    # st.set_page_config(layout="wide")

    # menucol1, menucol2, menucol3 = st.columns(3)

    # with menucol1:

        # pagesel = option_menu(None, ["Dashboard", "Advanced analytics", "Make a prediction"],
        #                         icons=['None', 'None', 'None'],
        #                         menu_icon="app-indicator", default_index=0, orientation="horizontal",
        #                         styles={
        #         "container": {"padding": "5!important", "background-color": "#f3e5e5", "color": "#8B0103"},
        #         "icon": {"color": "black", "font-size": "25px"}, 
        #         "nav-link": {"font-size": "12px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        #         "nav-link-selected": {"background-color": "#8B0103", "color": "#f3e5e5", "font-weight": "normal"},
        #     }
        #     )

    st.session_state['pagechoice'] = 'home'

    # if pagesel == "Dashboard":

        # value = sac.steps(
        # items=[
        #     sac.StepsItem(title='EDA', description='Explore'),
        #     sac.StepsItem(title='Overall', description='Entire dataset'),
        #     sac.StepsItem(title='Agency/Region', description='Grouped by area'),
        #     sac.StepsItem(title='Manager', description='Manager and below'),
        #     sac.StepsItem(title='Advisor', description='Individual data'),
        # ], format_func='title'
        # )

        # dbrdanalytics, dbrdmetrics, dbrdinteractive, dbrdneeds, dbrdpyg = st.tabs([":red[Analytics]", ":red[Metrics]", ":red[Interactive]", ":red[Needs assessment]", ":red[Pyg]"])

        # dbrdtype = TabBar(tabs=["Analytics","Tab2"],default=0,background = "white", color="red",activeColor="red",fontSize="14px")

        # import extra_streamlit_components as stx

        # chosen_id = stx.tab_bar(data=[
        #     stx.TabBarItemData(id=1, title="ToDo", description="Tasks to take care of"),
        #     stx.TabBarItemData(id=2, title="Done", description="Tasks taken care of"),
        #     stx.TabBarItemData(id=3, title="Overdue", description="Tasks missed out"),
        # ], default=1)

    col1, col2, col3 = st.columns([3.1,3,1])

    # col2.image("indalologo.png", width=250)

    col2.image("indalologo.jpg", width=180, caption="")

    coltree1, coltree2, coltree3 = st.columns([1,3,1])

    with coltree1:
            
        # Define the bullet bars data
        bullet_bars = [
            {"image_url": "https://image.pngaaa.com/370/2709370-middle.png", "text": "Initiating and supporting high impact, environmentally friendly, and social entrepreneurial innovations", "color": "#ffcccc"},  # Light red
            {"image_url": "https://image.pngaaa.com/370/2709370-middle.png", "text": "Fostering partnerships through dialogue among ecosystem stakeholders", "color": "#ff9999"},  # Medium red
            ]

        # Generate the HTML for the bullet bars
        bullet_bar_html = """
        <div style="width: 100%; margin: 0; padding: 0;"">
        """
        for bar in bullet_bars:
            bullet_bar_html += f"""
            <div style="display: flex; align-items: center; border: 1px solid #ccc; margin: 50px 0; padding: 10px; background-color: {bar['color']}; border-radius: 5px; width: 300px">
                <img src="{bar['image_url']}" style="width: 40px; height: 40px; margin-right: 10px; border-radius: 5px;">
                <span style="font-size: 18px; font-weight: normal; font-family: 'Roboto';">{bar['text']}</span>
            </div>
            """
        bullet_bar_html += "</div>"

        add_vertical_space(12)

        # Render the bullet bars in Streamlit
        st.components.v1.html(bullet_bar_html, height=300)

    # with coltree2:
    #     st.markdown(
    #         """
    #         <div style="text-align: center;">
    #             <img src="https://i.postimg.cc/439FCGRQ/growingtree.gif" width="300" />
    #         </div>
    #         """,
    #         unsafe_allow_html=True,
    #     )

    # from streamlit_carousel import carousel

    # # Define carousel items with 'img', 'title', and 'text'
    # test_items = [
    #     dict(
    #         img="./gui/images/socialentrep.png",
    #         title="Principle 1",
    #         text="Fostering social entrepreneurship",
    #     ),
    #     dict(
    #         img="https://img.freepik.com/free-photo/beautiful-wooden-pathway-going-breathtaking-colorful-trees-forest_181624-5840.jpg?w=1380&t=st=1688825780~exp=1688826380~hmac=dbaa75d8743e501f20f0e820fa77f9e377ec5d558d06635bd3f1f08443bdb2c1",
    #         title="Slide 2",
    #         text="A wooden bridge in a forest in Autumn",
    #     ),
    #     dict(
    #         img="https://img.freepik.com/free-photo/aerial-beautiful-shot-seashore-with-hills-background-sunset_181624-24143.jpg?w=1380&t=st=1688825798~exp=1688826398~hmac=f623f88d5ece83600dac7e6af29a0230d06619f7305745db387481a4bb5874a0",
    #         title="Slide 3",
    #         text="A distant mountain chain preceded by a sea",
    #     ),
    # ]

    # # Render the carousel
    # carousel(
    #     items=test_items,
    #     container_height=350,  # Adjust to fit image and text
    #     width=0.5,             # Adjust width
    #     pause=True,            # Pause on hover
    #     wrap=True,             # Enable slide wrapping
    #     fade=True,             # Enable fade transition
    #     interval=3000          # Time interval between slides in ms
    # )

    with coltree2:

        st.subheader("What is Veritas? This is next-generation, predictive Monitoring and Evaluation.")

        html_code="""        
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Interactive Quadrants</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background-color: #ffffff;
                    overflow: hidden;
                }

                .quadrant-container {
                    position: relative;
                    width: 500px;
                    height: 500px;
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    grid-template-rows: 1fr 1fr;
                    gap: 10px;
                }

                .quadrant {
                    position: relative;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    background-color: #f4f4f4;
                    border: 2px solid maroon;
                    border-radius: 10px;
                    opacity: 0;
                    transform: scale(0.8);
                    animation: quadrantFadeIn 1s forwards ease-out;
                    transition: transform 0.3s, box-shadow 0.3s;
                }

                .top-quadrant {
                    flex-direction: column; /* Text above the image */
                }

                .bottom-quadrant {
                    flex-direction: column-reverse; /* Text below the image */
                }

                .quadrant:hover {
                    transform: scale(1.05);
                    box-shadow: 0 0 20px rgba(255,128,128);
                }

                .quadrant img {
                    width: 200px;
                    height: 200px;
                    object-fit: contain;
                }

                .quadrant .quadrant-text {
                    margin-bottom: 10px;
                    font-size: 14px;
                    font-weight: bold;
                    text-align: center;
                }

                .center-logo {
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    width: 180px;
                    height: 120px;
                }

                .tooltip {
                    position: absolute;
                    background-color: #fff;
                    padding: 10px;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    font-size: 18px;
                    display: none;
                    white-space: nowrap;
                    pointer-events: none;
                }

                /* Animations */
                @keyframes quadrantFadeIn {
                    0% { opacity: 0; transform: scale(0.8); }
                    100% { opacity: 1; transform: scale(1); }
                }

                .quadrant:nth-child(1) {
                    animation-delay: 0.2s;
                }

                .quadrant:nth-child(2) {
                    animation-delay: 0.4s;
                }

                .quadrant:nth-child(3) {
                    animation-delay: 0.6s;
                }

                .quadrant:nth-child(4) {
                    animation-delay: 0.8s;
                }
            </style>
        </head>
        <body>

        <div class="quadrant-container">
            <div class="quadrant top-quadrant" data-tooltip="Analytical Insights, from deep data analysis.">
                <div class="quadrant-text">1. Analytical Insights</div>
                <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRO_lJMo867Kr-lkPCsDJuUWyn0cHBDsJ2KRg&s" alt="AI">
            </div>
            <div class="quadrant top-quadrant" data-tooltip="Domain Expertise, emphasizing industry knowledge.">
                <div class="quadrant-text">2. Domain Expertise</div>
                <img src="https://media.licdn.com/dms/image/D4D12AQFpNCyYnEwvVQ/article-cover_image-shrink_720_1280/0/1695042437428?e=2147483647&v=beta&t=sB1TIH7LyqWiORuex0f2llRW8nsLDep9haoZxsMUDXQ" alt="DE">
            </div>
            <div class="quadrant bottom-quadrant" data-tooltip="Governance and Monitoring and Evaluation, leveraging advanced algorithms.">
                <div class="quadrant-text">3. Governance and M&E</div>
                <img src="https://trialogueknowledgehub.co.za/wp-content/uploads/2019/08/monitoring-impact.jpg" alt="GV">
            </div>
            <div class="quadrant bottom-quadrant" data-tooltip="Planning & Optimization, from strategic vision.">
                <div class="quadrant-text">4. Planning & Optimization</div>
                <img src="https://wmep.org/wp-content/uploads/2022/05/iStock-1202205418-1024x540-1.jpg" alt="PO">
            </div>
            <img src="https://i.postimg.cc/Dy0sMXkR/veritaslogovert.png" alt="Logo" class="center-logo">
        </div>
        <div class="tooltip" id="tooltip"></div>

        <script>
            const quadrants = document.querySelectorAll('.quadrant');
            const tooltip = document.getElementById('tooltip');

            quadrants.forEach((quadrant) => {
                quadrant.addEventListener('mouseenter', (e) => {
                    const tooltipText = quadrant.getAttribute('data-tooltip');
                    tooltip.textContent = tooltipText;
                    tooltip.style.display = 'block';

                    const rect = quadrant.getBoundingClientRect();
                    tooltip.style.left = `${rect.x + rect.width / 2 - 100}px`;
                    tooltip.style.top = `${rect.y - 30}px`;
                });

                quadrant.addEventListener('mouseleave', () => {
                    tooltip.style.display = 'none';
                });
            });
        </script>

        </body>
        </html>
        """

        # Embed in Streamlit
        st.components.v1.html(html_code, height=700, scrolling=False)

    with coltree3:

        add_vertical_space(12)
            
        # Define the bullet bars data
        bullet_bars = [
            {"image_url": "https://image.pngaaa.com/370/2709370-middle.png", "text": "Building a strong, robust institution with good goernance, excellent team, good citizenry", "color": "#ffcccc"},  # Medium red
            {"image_url": "https://image.pngaaa.com/370/2709370-middle.png", "text": "Policy facilitation, co-creation of new policies, adaptation of new policies", "color": "#ff9999"},  # Darker red
        ]

        # Generate the HTML for the bullet bars
        bullet_bar_html = """
        <div style="width: 100%; margin: 0; padding: 0;">
        """
        for bar in bullet_bars:
            bullet_bar_html += f"""
            <div style="display: flex; align-items: center; border: 1px solid #ccc; margin: 50px 0; padding: 10px; background-color: {bar['color']}; border-radius: 5px;">
                <img src="{bar['image_url']}" style="width: 40px; height: 40px; margin-right: 10px; border-radius: 5px;">
                <span style="font-size: 18px; font-weight: normal; font-family: "Arial";">{bar['text']}</span>
            </div>
            """
        bullet_bar_html += "</div>"

        # Render the bullet bars in Streamlit
        st.components.v1.html(bullet_bar_html, height=300)