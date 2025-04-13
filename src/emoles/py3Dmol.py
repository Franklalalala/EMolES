import py3Dmol
import os


def cube_2_html(cube_path, iso_value):
    # Read the cube file
    with open(cube_path, 'r') as f:
        cube_data = f.read()
    # Create visualization
    view = py3Dmol.view()
    view.addModel(cube_data, 'cube')
    view.addVolumetricData(cube_data, "cube", {'isoval': -iso_value, 'color': "red", 'opacity': 0.75})
    view.addVolumetricData(cube_data, "cube", {'isoval': iso_value, 'color': "blue", 'opacity': 0.75})
    view.setStyle({'stick': {}})
    view.zoomTo()
    # Generate HTML content
    html_content = view._make_html()
    # Create HTML file with the same name as the cube file
    html_path = os.path.splitext(cube_path)[0] + '.html'
    with open(html_path, "w") as out:
        out.write(html_content)

    print(f"Generated HTML for: {cube_path}")


def cubes_2_htmls(out_path, iso_value):
    for dirpath, dirnames, filenames in os.walk(out_path):
        for filename in filenames:
            if filename.endswith('.cube'):
                cube_path = os.path.join(dirpath, filename)
                cube_2_html(cube_path, iso_value)
