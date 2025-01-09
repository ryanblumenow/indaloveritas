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
        
            html_code="""
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
                        id: 'doughnutLabels',
                        afterDatasetsDraw(chart) {
                            const {ctx, data} = chart;
                            chart.getDatasetMeta(0).data.forEach((datapoint, index) => {
                                const {x, y} = datapoint.tooltipPosition();
                                ctx.fillStyle = '#000';
                                ctx.font = 'bold 16px Arial'; // Adjusted font size for better fit
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'middle';
                                const lines = data.labels[index].split(' '); // Split label into multiple lines
                                lines.forEach((line, i) => {
                                    ctx.fillText(line, x, y - 10 + i * 15); // Adjust line height
                                });
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