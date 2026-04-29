
        from pymol import cmd,stored
        
        set depth_cue, 1
        set fog_start, 0.4
        
        set_color b_col, [36,36,85]
        set_color t_col, [10,10,10]
        set bg_rgb_bottom, b_col
        set bg_rgb_top, t_col      
        set bg_gradient
        
        set  spec_power  =  200
        set  spec_refl   =  0
        
        load "data/2pn7.pdb", protein
        create ligands, protein and organic
        select xlig, protein and organic
        delete xlig
        
        hide everything, all
        
        color white, elem c
        color bluewhite, protein
        #show_as cartoon, protein
        show surface, protein
        #set transparency, 0.15
        
        show sticks, ligands
        set stick_color, magenta
        
        
        
        
        # SAS points
 
        load "data/2pn7.pdb_points.pdb.gz", points
        hide nonbonded, points
        show nb_spheres, points
        set sphere_scale, 0.2, points
        cmd.spectrum("b", "green_red", selection="points", minimum=0, maximum=0.7)
        
        
        stored.list=[]
        cmd.iterate("(resn STP)","stored.list.append(resi)")    # read info about residues STP
        lastSTP=stored.list[-1] # get the index of the last residue
        hide lines, resn STP
        
        cmd.select("rest", "resn STP and resi 0")
        
        for my_index in range(1,int(lastSTP)+1): cmd.select("pocket"+str(my_index), "resn STP and resi "+str(my_index))
        for my_index in range(1,int(lastSTP)+1): cmd.show("spheres","pocket"+str(my_index))
        for my_index in range(1,int(lastSTP)+1): cmd.set("sphere_scale","0.4","pocket"+str(my_index))
        for my_index in range(1,int(lastSTP)+1): cmd.set("sphere_transparency","0.1","pocket"+str(my_index))
        
        
        
        set_color pcol1 = [0.361,0.576,0.902]
select surf_pocket1, protein and id [88,93,1006,1008,141,142,402,403,404,411,413,85,448,441,445,444,449,729,388,384,387,389,683,94,100,101,684,68,72,78,84,670,671,175,176,674,740,891,739,741,80,81,82,727] 
set surface_color,  pcol1, surf_pocket1 
set_color pcol2 = [0.365,0.278,0.702]
select surf_pocket2, protein and id [1754,1484,1485,1746,1756,1480,1519,1787,1784,2064,1731,2341,2343,1427,1428,1431,1436,1443,1421,1791,1788,1792,2027,2013,2014,2017,2074,2076,2026,2015,2226,1411,1415,1444,2225,1423,1425,2075] 
set surface_color,  pcol2, surf_pocket2 
set_color pcol3 = [0.792,0.361,0.902]
select surf_pocket3, protein and id [910,916,918,1812,1815,2273,1813,2245,2251,2271,2253,1803,928,938,2275,1817,2268] 
set surface_color,  pcol3, surf_pocket3 
set_color pcol4 = [0.702,0.278,0.533]
select surf_pocket4, protein and id [930,936,938,940,933,939,951,473,324,474,475,956,910,916,917,918,2273,923,2251,469,472,2253,470,471,2263] 
set surface_color,  pcol4, surf_pocket4 
set_color pcol5 = [0.902,0.361,0.361]
select surf_pocket5, protein and id [2210,772,2213,1969,1970,2090,769,770,2104,2209,2107,627,875,878,874,876,626,866,29,867,863] 
set surface_color,  pcol5, surf_pocket5 
set_color pcol6 = [0.702,0.533,0.278]
select surf_pocket6, protein and id [2302,2303,1773,2553,2556,1724,1725,1765,1690,2296,1776] 
set surface_color,  pcol6, surf_pocket6 
   
        
        deselect
        
        orient
        