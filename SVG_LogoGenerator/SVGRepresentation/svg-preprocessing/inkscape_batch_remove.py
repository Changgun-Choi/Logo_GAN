import os


def run_command(command):
	# print(command)
	os.system(command)


def inkscape_remove_transform(svg_to_process, processed_file_path):
	# required to fix svgs created with an old inkscape version which used a different dpi setting
	run_command("inkscape -l --convert-dpi-method=scale-viewbox \"" + svg_to_process + "\" --export-filename=\"" + processed_file_path + "\"")
	# apply all transforms and remove them
	run_command("inkscape -g --batch-process --verb com.klowner.filter.apply_transform \"" + processed_file_path + "\" --export-filename=\"" + processed_file_path + "\"")
