from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Create the main layout for the screen
        main_layout = BoxLayout(orientation='vertical')

        # Create a layout for the navigation buttons
        nav_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50)

        # Create navigation buttons
        open_camera_button = Button(text="Open Camera")
        inventory_button = Button(text="Inventory")
        generate_recipe_button = Button(text="Generate Recipe")

        # Bind button events to functions
        open_camera_button.bind(on_release=self.change_screen_open_camera)
        inventory_button.bind(on_release=self.change_screen_inventory)
        generate_recipe_button.bind(on_release=self.change_screen_generate_recipe)

        # Add buttons to the navigation layout
        nav_layout.add_widget(open_camera_button)
        nav_layout.add_widget(inventory_button)
        nav_layout.add_widget(generate_recipe_button)

        # Add the navigation layout to the main layout
        main_layout.add_widget(nav_layout)

        # Add the main layout to the screen
        self.add_widget(main_layout)

    def change_screen_open_camera(self, *args):
        print("Button Pressed: Navigating to OpenCameraScreen")  # Debug print
        self.manager.current = "open_camera_screen"

    def change_screen_inventory(self, *args):
        print("Navigating to InventoryScreen")  # Debug print
        self.manager.current = "inventory_screen"

    def change_screen_generate_recipe(self, *args):
        print("Navigating to GenerateRecipeScreen")  # Debug print
        self.manager.current = "generate_recipe_screen"
