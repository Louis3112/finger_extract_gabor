filtered = self.coherence_diffusion_filter(segmented, filter_params["sigma"])
            
            # Step 4: Apply log gabor filter
            gabor_filtered = self.log_gabor_filter(filtered, 
                                                 wavelength=filter_params["lambda"], 
                                                 orientation=filter_params["theta"])