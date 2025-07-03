# pyspread/context.py

main_window_instance = None

def set_main_window_instance(instance):
    global main_window_instance
    main_window_instance = instance

def get_main_window_instance():
    return main_window_instance