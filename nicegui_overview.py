from nicegui import ui
import pandas as pd

# Text
ui.label("This is a label")
# Markdown
ui.markdown("### This is markdown")
# HTML
ui.html("<p>This is a p tag</p>")
# Mermaid
ui.mermaid("""
graph TD;
    A-->B;
    A-->C;
    B-->D;
    C-->D;
""")

# Widgets
# Text input
ui.input(label="First name", on_change= lambda e: fname.set_text(e.value))
fname = ui.label()
# Text area
ui.textarea(label="Message", on_change= lambda e: msg.set_text(e.value))
msg = ui.label()

# Number
ui.number(label="Age", min=2, max=100)

# Date
ui.date(value="2023-10-03")

# Time
ui.time(value="12:00")

# Color
ui.color_input()
#ui.color_picker()

# Button
ui.button(text="Click me", on_click=lambda e: ui.notify("You clicked the button"))
radio1 = ui.radio([1,2,5], value=1).props("inline")
select_gender = ui.select(["Male", "Female"])
# Checkbox
checkbox = ui.checkbox("Check me")
# Switch
switch_btn = ui.switch("Switch")

# Slider
salary = ui.slider(min=100, max=10000, value=150)

# Media
# Image
# ui.image(source="retriever.jpg")
# Audio
# ui.audio()
# Video
# ui.video(source="xxx.mp4")

# Data and table
df = pd.read_csv("exampleforecastdata.csv")
ui.aggrid.from_pandas(df)

# ui.table()

# Code
ui.code("""
	df = pd.read_csv("exampleforecastdata.csv")
ui.aggrid.from_pandas(df)
""", language="python")

# ui.json_editor()
ui.editor()

# Message and chat element
ui.chat_message("Hello, this is NiceGUI")

# Separator
ui.separator()

# Tabs
with ui.tabs().classes("w-full") as tabs:
    home = ui.tab("Home tab")
    about = ui.tab("About tab")

# Assign or group some elements under each tab
with ui.tab_panels(tabs, value=home).classes("w-full"):
    with ui.tab_panel(home):
        ui.label("This is the home tab")
        ui.textarea(label="Items to do")

    with ui.tab_panel(about):
        ui.label("This is the about tab")
        ui.textarea(label="Items to do")

# Pages
@ui.page("/blog")
def blog():
    ui.label("This is the blog page")

@ui.page("/blog/{id}")
def blog(id:int):
    ui.label(f"This is the blog page {id}")

ui.link("Blog", blog)
ui.link("Blog", "blog/3")

# Card
with ui.card().tight():
    ui.image(source="retriever.jpg")
    with ui.card_section():
        ui.label("This is a card")
    with ui.card_actions():
        ui.button("This is a button")

# File upload
ui.upload(label="Upload csv")
ui.download(src="retriever.jpg")

# Dark mode
dark = ui.dark_mode()
ui.button("Dark", on_click=dark.enable)
ui.button("Light", on_click=dark.disable)

# Binding
class Todo:
    def __init__(self) -> None:
        self.number = 1
        self.task = ""

todo = Todo()
ui.input("Task").bind_value(todo, "task")
ui.slider(min=4, max=10).bind_value(todo, "number")
# Show
ui.label().bind_text(todo, "task")
ui.label().bind_text(todo, "number")

ui.run(host="127.0.0.1", port=8001)
