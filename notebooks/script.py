import ipywidgets as widgets

form_item_layout = Layout(
    display='flex',
    flex_flow='row',
    justify_content='space-between',
    
)

arrival_hour = widgets.BoundedIntText( min = 8, max = 16, step = 1, disabled = False,\
                                    layout = form_item_layout)
arrival_minute = widgets.BoundedIntText(value = 30,min = 0,max = 59, step = 1, disabled = False, layout =form_item_layout)

start_stop = widgets.Dropdown(options=relevant_nodes.select("stop_name"), layout=form_item_layout)

end_stop =  widgets.Dropdown(options=relevant_nodes.select("stop_name"), layout=form_item_layout)

valid = widgets.Button(description='Search', disabled=False, button_style='primary',tooltip='Go ahead and click',icon='search')


out = widgets.Output()
valid.on_click(search)


hbox1 = widgets.HBox([widgets.Label(value="From:", layout=form_item_layout),start_stop,widgets.Label(value="To:", layout=form_item_layout), end_stop])
hbox2 = widgets.HBox([widgets.Label(value="Arrives at:",), arrival_hour, arrival_minute])

ui = widgets.VBox([ hbox1, hbox2])
display(ui, out)
