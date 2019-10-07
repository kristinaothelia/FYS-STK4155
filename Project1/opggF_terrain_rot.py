		# Load the terrain
		#terrain_image = imread('Oslo.tif')
		terrain_image = imread('PIA23328.tif')
		#terrain_image = imread('MAUNA_KEA.tif')
		#reduced_image = terrain_image[200:1000, 700:1500] # Oslo
		reduced_image = terrain_image[700:1100, 200:600]  # Mars
		#reduced_image = terrain_image[100:1200, 1300:2400]

		p_degree      = 5
		
		#terrain_arr = np.array(reduced_image)
		terrain_arr   = np.array(terrain_image)
		n_x           = len(terrain_arr)
		print(terrain_arr.shape)

		x             = np.arange(0, terrain_arr.shape[1])/(terrain_arr.shape[1]-1)
		y             = np.arange(0, terrain_arr.shape[0])/(terrain_arr.shape[0]-1)
		x,y           = np.meshgrid(x,y)

		
		plt.figure(1)
		#project1_plot.plot_terrain(x,y,terrain_image, "Terrain_original", func="Original", string='Oslofjord', savefig=False)
		project1_plot.plot_terrain(x,y,terrain_image, "Terrain_original", func="Original", string='Utopia Planitia, Mars', savefig=True)
		plt.figure(2)
		#project1_plot.plot_terrain(x,y,reduced_image, "Terrain_cropped", func="Original", string='Inner Oslofjord', savefig=False)
		project1_plot.plot_terrain(x,y,reduced_image, "Terrain_cropped", func="Original", string='Unnamed crater in Utopia Planitia, Mars', savefig=True)

		final_image = cv2.resize(reduced_image, dsize=(200, 200), interpolation=cv2.INTER_NEAREST)
		x_          = np.arange(0, final_image.shape[1])/(final_image.shape[1]-1)
		y_          = np.arange(0, final_image.shape[0])/(final_image.shape[0]-1)
		x_,y_       = np.meshgrid(x_,y_)

		print(final_image.shape)

		plt.figure(3)
		#project1_plot.plot_terrain(x_,y_,final_image, "Terrain_final", func="Original", string='Inner Oslofjord final', savefig=False)
		project1_plot.plot_terrain(x_,y_,final_image, "Terrain_final", func="Original", string='Unnamed crater in Utopia Planitia, Mars', savefig=True)

		plt.figure(4)
		project1_plot.plot_3D(x_, y_,final_image, p_degree, "3D_terrain_real_data", "Final_terrain", savefig=True)
		plt.show()
		
		sys.exit()