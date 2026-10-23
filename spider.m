folder = fileparts(mfilename('fullpath'));
py.sys.path.append(folder);

output = py.genetic_algorithm_V2.main();
plot_spider_pose(output)