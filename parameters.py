arguments = {"nb_classes" : 3,
             "nb_RNlayers" : 50,
             "isDilation" : True,
             "upscale" : 8,
             "DropOutDecoder" : 1, 
             "FeatureDropDecoder" : 1, 
             "FeatureNoiseDecoder" : 1, 
             "VATDecoder" : 1, 
             "CutOutDecoder" : 1, 
             "ContextMaskingDecoder" : 1, 
             "ObjectMaskingDecoder" : 1, 
             "xi" : 1e-1,
             "eps" : 10.0, 
             "drop_rate" : 0.3, 
             "spacial_dropout" : True,
             "erase" : 0.4,
             "uniform_range" : 0.3}

aux_decoder_order = {"DropOutDecoder" : 0, 
                    "FeatureDropDecoder" : 1, 
                    "FeatureNoiseDecoder" : 2, 
                    "VATDecoder" : 3, 
                    "CutOutDecoder" : 4, 
                    "ContextMaskingDecoder" : 5, 
                    "ObjectMaskingDecoder" : 6                     
}