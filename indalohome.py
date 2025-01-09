import streamlit as st

#add an import to Hydralit
from hydralit import HydraHeadApp
from hydralit import HydraApp

#create a wrapper class
class indalohome(HydraHeadApp):

#wrap all your code in this method and you should be done

    def run(self):

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

        from st_on_hover_tabs import on_hover_tabs

        st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)

        with st.sidebar:
                st.image('indalologo.jpg')
                add_vertical_space(1)
                value = on_hover_tabs(tabName=['KPI Overview', 'Dashboard/Analytics'], 
                                    iconName=['contacts', 'dashboard', 'account_tree', 'table', 'report', 'edit', 'update', 'pivot_table_chart', 'menu_book'],
                                    styles = {'navtab': {'background-color':'#6d0606',
                                                        'color': 'white',
                                                        'font-size': '18px',
                                                        'transition': '.3s',
                                                        'white-space': 'nowrap',
                                                        'text-transform': 'uppercase'},
                                            'tabOptionsStyle': {':hover :hover': {'color': 'red',
                                                                            'cursor': 'pointer'}},
                                            'iconStyle':{'position':'fixed',
                                                            'left':'7.5px',
                                                            'text-align': 'left'},
                                            'tabStyle' : {'list-style-type': 'none',
                                                            'margin-bottom': '30px',
                                                            'padding-left': '30px'}},
                                    key="hoversidebar",
                                    default_choice=0)

        css = '''
        <style>
            .stTabs [data-baseweb="tab-highlight"] {
                background-color:blue;
            }
        </style>
        '''

        st.markdown(css, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([3.1,3,1])

        col2.image("veritaslogo.png", width=250)

        # col2.image("indalologo.jpg", use_column_width=False, width=180, caption="")

        coltree1, coltree2, coltree3 = st.columns([1,3,1])

        with coltree1:
             
            # Define the bullet bars data
            bullet_bars = [
                {"image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS8px86qz5KCA0O2TieLj6u1ecar5d229kDNw&s", "text": "Initiating and supporting high impact, environmentally friendly, and social entrepreneurial innovations", "color": "#ffcccc"},  # Light red
                {"image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS8px86qz5KCA0O2TieLj6u1ecar5d229kDNw&s", "text": "Fostering oartnerships through dialogue among ecosystem stakeholders", "color": "#ff9999"},  # Medium red
                ]

            # Generate the HTML for the bullet bars
            bullet_bar_html = """
            <div style="width: 100%; margin: 0; padding: 0;">
            """
            for bar in bullet_bars:
                bullet_bar_html += f"""
                <div style="display: flex; align-items: center; border: 1px solid #ccc; margin: 50px 0; padding: 10px; background-color: {bar['color']}; border-radius: 5px; width: 300px">
                    <img src="{bar['image_url']}" style="width: 40px; height: 40px; margin-right: 10px; border-radius: 5px;">
                    <span style="font-size: 16px; font-weight: normal; font-family: 'Arial';">{bar['text']}</span>
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

            st.subheader("What is Veritas? This is next-generation, predictive Monitoring and Evaluation")

            html_code = """        
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Interactive Donut Chart</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        margin: 0;
                        background-color: #ffffff;
                        position: relative;
                        overflow-y: auto;
                        height: 100vh;
                    }

                    canvas {
                        width: 80%;
                        height: auto;
                    }

                    .info-box {
                        position: absolute;
                        background: #fff;
                        padding: 15px;
                        border-radius: 10px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        font-size: 18px;
                        font-weight: bold;
                        color: #333;
                        width: 500px;
                        text-align: center;
                        display: none;
                        white-space: nowrap;
                        pointer-events: none; /* Prevent mouse interaction */
                    }

                    .center-image {
                        position: absolute;
                        width: 260px;
                        height: 150px;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        border-radius: 0%;
                    }
                </style>
            </head>
            <body>

            <canvas id="donutChart" width="600" height="600"></canvas>
            <img src="https://i.postimg.cc/pXBhPcHN/indalologo.png" alt="Cognitive Analytics" class="center-image">
            <div class="info-box" id="infoBox">Hover over a segment</div>

            <script>
                const ctx = document.getElementById('donutChart').getContext('2d');
                const infoBox = document.getElementById('infoBox');
                let currentHoverIndex = null; // Track the currently hovered segment
                let hideTooltipTimeout = null; // Timeout for hiding the tooltip

                const data = {
                    labels: [
                        '2. Analytical Insights', 
                        '3. Domain Expertise', 
                        '4. Governance', 
                        '1. Planning and optimization'
                    ],
                    datasets: [{
                        data: [25, 25, 25, 25],
                        backgroundColor: [
                            'rgba(255, 223, 186, 0.9)', 
                            'rgba(186, 225, 255, 0.9)', 
                            'rgba(186, 255, 240, 0.9)', 
                            'rgba(255, 204, 204, 0.9)'
                        ],
                        borderWidth: 1,
                        hoverOffset: 20,
                    }]
                };

                const descriptions = [
                    "Analytical Insights, from deep data analysis.",
                    "Domain Expertise, emphasizing industry knowledge.",
                    "Monitoring and Evaluation, leveraging advanced algorithms.",
                    "Strategic Vision, guiding long-term planning."
                ];

                const quadrantImages = [
                    'data:image/webp;base64,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', // Replace with your image URLs
                    'https://example.com/image2.png',
                    'https://example.com/image3.png',
                    'https://example.com/image4.png'
                ];

                const loadedImages = [];
                quadrantImages.forEach((src, index) => {
                    const img = new Image();
                    img.src = src;
                    loadedImages[index] = img;
                });

                const config = {
                    type: 'doughnut',
                    data: data,
                    options: {
                        responsive: false,
                        plugins: {
                            legend: {
                                display: false, // Hide the legend
                            },
                            tooltip: {
                                enabled: false, // Disable default tooltips
                            }
                        },
                        onHover: (event, elements) => {
                            clearTimeout(hideTooltipTimeout); // Clear the hide timeout
                            if (elements.length > 0) {
                                const segmentIndex = elements[0].index; // Index of the hovered segment

                                // Update the tooltip only if a new segment is hovered
                                if (currentHoverIndex !== segmentIndex) {
                                    currentHoverIndex = segmentIndex; // Update current hover index
                                    const { x, y } = elements[0].element.tooltipPosition(); // Tooltip position
                                    const chartCenterX = event.chart.width / 2; // Horizontal center of the chart
                                    const chartCenterY = event.chart.height / 2; // Vertical center of the chart

                                    // Ensure infoBox stays within visible area
                                    const bodyRect = document.body.getBoundingClientRect();
                                    const infoBoxWidth = infoBox.offsetWidth;
                                    const infoBoxHeight = infoBox.offsetHeight;

                                    // Determine the quadrant
                                    let left, top;
                                    if (x >= chartCenterX && y <= chartCenterY) {
                                        left = x + 100;
                                        top = y - infoBoxHeight / 2;
                                    } else if (x < chartCenterX && y <= chartCenterY) {
                                        left = x - infoBoxWidth - 100;
                                        top = y - infoBoxHeight / 2;
                                    } else if (x < chartCenterX && y > chartCenterY) {
                                        left = x - infoBoxWidth - 100;
                                        top = y - infoBoxHeight / 2;
                                    } else {
                                        left = x + 100;
                                        top = y - infoBoxHeight / 2;
                                    }

                                    // Ensure the tooltip stays within the viewport
                                    if (left + infoBoxWidth > bodyRect.width) {
                                        left = bodyRect.width - infoBoxWidth - 10;
                                    }
                                    if (left < 0) {
                                        left = 10;
                                    }
                                    if (top + infoBoxHeight > bodyRect.height) {
                                        top = bodyRect.height - infoBoxHeight - 10;
                                    }
                                    if (top < 0) {
                                        top = 10;
                                    }

                                    // Set the position and show the tooltip
                                    infoBox.style.left = `${left}px`;
                                    infoBox.style.top = `${top}px`;
                                    infoBox.style.display = 'block';
                                    infoBox.textContent = descriptions[segmentIndex];
                                }
                            } else {
                                if (currentHoverIndex !== null) {
                                    currentHoverIndex = null; // Reset hover index
                                    hideTooltipTimeout = setTimeout(() => {
                                        infoBox.style.display = 'none'; // Hide the tooltip
                                    }, 50); // Small delay to prevent flickering
                                }
                            }
                        },
                    },
                    plugins: [{
                        id: 'quadrantImages',
                        afterDraw(chart) {
                            const { ctx, chartArea, width, height } = chart;
                            const meta = chart.getDatasetMeta(0);
                            const radius = (meta.data[0].outerRadius + meta.data[0].innerRadius) / 2;

                            meta.data.forEach((datapoint, index) => {
                                const { x, y } = datapoint.tooltipPosition();
                                const angle = datapoint.startAngle + (datapoint.endAngle - datapoint.startAngle) / 2;

                                // Calculate image position (place slightly outward from center of each segment)
                                const imgX = chartArea.left + (width / 2) + radius * Math.cos(angle);
                                const imgY = chartArea.top + (height / 2) + radius * Math.sin(angle);

                                const img = loadedImages[index];
                                if (img.complete) {
                                    ctx.drawImage(img, imgX - 15, imgY - 15, 30, 30); // Draw image (adjust size as needed)
                                }
                            });
                        }
                    }]
                };

                new Chart(ctx, config);
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
                {"image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS8px86qz5KCA0O2TieLj6u1ecar5d229kDNw&s", "text": "Building a strong, robust institution with good goernance, excellent team, good citizenry", "color": "#ffcccc"},  # Medium red
                {"image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS8px86qz5KCA0O2TieLj6u1ecar5d229kDNw&s", "text": "Policy facilitation, co-creation of new policies, adaptation of new policies", "color": "#ff9999"},  # Darker red
            ]

            # Generate the HTML for the bullet bars
            bullet_bar_html = """
            <div style="width: 100%; margin: 0; padding: 0;">
            """
            for bar in bullet_bars:
                bullet_bar_html += f"""
                <div style="display: flex; align-items: center; border: 1px solid #ccc; margin: 50px 0; padding: 10px; background-color: {bar['color']}; border-radius: 5px;">
                    <img src="{bar['image_url']}" style="width: 40px; height: 40px; margin-right: 10px; border-radius: 5px;">
                    <span style="font-size: 16px; font-weight: normal; font-family: "Arial";">{bar['text']}</span>
                </div>
                """
            bullet_bar_html += "</div>"

            # Render the bullet bars in Streamlit
            st.components.v1.html(bullet_bar_html, height=300)