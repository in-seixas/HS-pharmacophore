from pymol import cmd
from drugpy.ftmap.core import load_atlas
from drugpy.commons import fo_ 
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
from pharmacophore import InteractionKind, Feature, PharmacophoreJsonWriter
import matplotlib.cm as cm
from sklearn import metrics
from scipy import stats
from scipy.spatial import distance
from glob import glob


#pegue o sítio ativo
def get_active_site(caminho_ftmap:str, saida_active_site: str ):

    ftmap = load_atlas(caminho_ftmap, plot = False, table = False)

    objeto = list(ftmap.keys())[0]
    print(objeto)
    
    hotspots, clusters  = ftmap[objeto]

    all_hotspots = []
    join_hs_D = []
 
    for hotspot in hotspots:
        
        all_hotspots.append(
            {
            "Selection": hotspot.selection, 
            "Strenght": hotspot.strength,
            "Klass": hotspot.klass  })

    hs_D = list(filter(lambda x: x["Klass"] == "D", all_hotspots))
    
    hs_D.sort(key= lambda x : x["Strenght"], reverse= True)

    hotspot_max = list(hs_D[0].values())[0]   #hs mais forte

    for dict_hs in hs_D:
        
        fo = fo_(hotspot_max, dict_hs["Selection"])
        
        if fo >= 0.5: 
                
            join_hs_D.append(dict_hs["Selection"])

    
    all_druggable = " or ".join(join_hs_D)
      
    print("ALL DRUGGABLE", all_druggable) 

    cmd.create("all_druggable", all_druggable)
                
    cmd.select("active_site", "byres polymer within 8 of all_druggable")
    
    cmd.save(saida_active_site, "active_site", format="pdb")

    cmd.reinitialize()
        

#get coord of hotspots
def get_coord_of_hotspot(caminho_ftmap:str):

    ftmap = load_atlas(caminho_ftmap, plot = False, table = False)

    objeto = list(ftmap.keys())[0]
    
    hotspots, clusters  = ftmap[objeto]

    hotspots_druggables = []
    join_hs_D = []
 
    for hotspot in hotspots:
        
        hotspots_druggables.append(
            {
            
            "Selection": hotspot.selection, 
            "Strenght": hotspot.strength,
            "Klass": hotspot.klass  })

    hs_D = list(filter(lambda x: x["Klass"] == "D", hotspots_druggables))
    
    hs_D.sort(key= lambda x : x["Strenght"], reverse= True)

    hotspot_max = list(hs_D[0].values() )[0]  #hs mais forte

    for dict_hs in hs_D:
        
        fo = fo_(hotspot_max, dict_hs["Selection"])
        
        if fo >= 0.5: 
                
            join_hs_D.append(dict_hs["Selection"])

        
    all_druggable = " or ".join(join_hs_D)
  

    cmd.create("all_druggable", all_druggable)
                
    coorOfHotspot = cmd.get_extent("all_druggable")


    return coorOfHotspot,  "all_druggable"




def get_features_from_ftmap_and_fragmap(caminho_ftmap:str,
                                        caminho_fragmap:str, 
                                        caminho_dump:str, 
                                        tipo:str, 
                                        level_param:int,
                                        ):

    coorOfHotspot, hotspot_sel  = get_coord_of_hotspot(caminho_ftmap)


    x_min = coorOfHotspot[0][0]
    x_max = coorOfHotspot[1][0]
    y_min = coorOfHotspot[0][1]
    y_max = coorOfHotspot[1][1]
    z_min = coorOfHotspot[0][2]
    z_max = coorOfHotspot[1][2]
    

    cmd.load(caminho_fragmap, partial = 1)
    cmd.dump(f'{caminho_dump}/{tipo}.txt', tipo)
    

    level = []
    coordenadas = []

    with open(f'{caminho_dump}/{tipo}.txt', 'r') as r:
        for linha in r:
            x, y, z, level = map(float, linha.split())

            if x >  x_min and x < x_max and y > y_min and y <  y_max and z > z_min and z < z_max and level >= level_param :
                
                coordenadas.append((level, x, y, z))
    
    xyz = []
    levels = []

    for index, _ in enumerate(coordenadas):

        levels.append(coordenadas[index][0])
        xyz.append(coordenadas[index][1:4])
    

    return xyz, levels



def metrics_clusters(X, saida: str, tipo:str, level_map:int):

    sils = []
    chs = []
    dbs = []


    size = range(2, 6)

    for k in size:

        k2 = KMeans(random_state= 0, n_clusters= k)
        k2.fit(X)
        sils.append(

            metrics.silhouette_score(X, k2.labels_)

        )
        chs.append(
            metrics.calinski_harabasz_score(X, k2.labels_)
        )

        dbs.append(metrics.davies_bouldin_score(X, k2.labels_))

    fig, ax = plt.subplots(figsize=(6,4))

    (
        pd.DataFrame(
            {
            
                "silhouete":sils,
                "calinski":chs,
                "davis":dbs,
                "k":size,

            }
        )
        .set_index("k")
        .plot(ax=ax, subplots=True, layout=(2,2))
    )

    fig.savefig(f"{saida}/metrics_to_{tipo}_{level_map}.png", dpi= 300)    



def points_clusters(points, level, k:int, radius_mult):


    kmeans = KMeans(n_clusters = k, random_state=0).fit(points)
    centerofmass = kmeans.cluster_centers_
    centroides = centerofmass
    labels = kmeans.labels_ 
    
    df = pd.DataFrame(points, columns=["X", "Y", "Z"])
    df.loc[:, "Level"] = level
    df.loc[:, "Labels"] = labels
            
    points_and_radius = []
    
    for n in range(0, len(centroides)):
       
        #sum countours of points fragmap in each cluster
        max_contours = df.groupby(['Labels'])['Level'].sum()


        df_k = df.loc[df['Labels'] == n]  
     
        distance_ = []

    
        for row in df_k.iterrows():

            
            xyz = list(row[1][0:3])

            #calculete distance of each point to it centroid in the cluster
            distance_.append(float(distance.euclidean(xyz, centroides[n])))

        
        points_and_radius.append((centroides[n], statistics.mean(distance_*radius_mult), max_contours[n]))


   
    return points_and_radius





def get_centroids_for_pharmacophore(caminho_ftmap: str, 
                            k_acceptor: int, 
                            k_donor: int, 
                            k_apolar:int, 
                            level:int):

    _, hs = get_coord_of_hotspot(caminho_ftmap)

    xyz_acceptor, level_acceptor = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], "acceptor", level)
    xyz_donor, level_donor = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], "donor", level)
    xyz_apolar, level_apolar = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], "apolar", level)

    clusters_acceptor = points_clusters(xyz_acceptor, level_acceptor, k_acceptor, 1)
    clusters_donor = points_clusters(xyz_donor, level_donor, k_donor, 1)
    clusters_apolar = points_clusters(xyz_apolar, level_apolar, k_apolar, 1)

    centers_acceptor = []

    for (x ,y, z), radius, level in clusters_acceptor:
        


        dca = cmd.count_atoms(f' name N*+O* and x > {x-2} and x < {x+2}' 
                           f' and y > {y-2} and y < {y+2} and z > {z-2} and z < {z+2} and {hs}')


        centers_acceptor.append({
            "X, Y, Z":(x,y,z),
            "TIPO":"ACCEPTOR",
            "DC": dca,
            "Raio":radius,
            "Level":level,
            "Level2":level/dca

        })       

    centers_donor = []

    for (x ,y, z), radius, level in clusters_donor:

        dcd = cmd.count_atoms(f'name N*+O* and x > {x-2} and x < {x+2}' 
                           f' and y > {y-2} and y < {y+2} and z > {z-2} and z < {z+2} and {hs}')

        centers_donor.append({
            "X, Y, Z":(x,y,z),
            "TIPO":"DONOR",
            "DC": dcd,
            "Raio":radius,
            "Level":level,
            "Level2":level/dcd

        })
    

    centers_apolar = []

    for (x ,y, z), radius, level in clusters_apolar:


        dch = cmd.count_atoms(f' name C* and x > {x-2} and x < {x+2}' 
                           f' and y > {y-2} and y < {y+2} and z > {z-2} and z < {z+2} and {hs}')

        centers_apolar.append({
            "X, Y, Z":(x,y,z),
            "TIPO":"HYDROFOBIC",
            "DC": dch,
            "Raio":radius,
            "Level":level,
            "Level2":level/dch
        })

    centers_model = list(centers_donor + centers_acceptor + centers_apolar)
    

    #ordenate features by DC
    centers_model.sort(key= lambda x : x["Level"], reverse= True)


    return centers_model
    
    
    
def build_pharmacohore(caminho_ftmap: str,
                        caminho_saida:str,
                        k_acceptor: int, 
                        k_donor: int, 
                        k_apolar:int, 
                        level:int):
    
    
    centroids = get_centroids_for_pharmacophore(caminho_ftmap, k_acceptor, k_donor, k_apolar, level)
    
    feats = []

    pharmacophore_writer = PharmacophoreJsonWriter()

    for dicts in centroids:
        for key, values in dicts.items():


            if dicts[key] == "ACCEPTOR":
    
                x = list(dicts.values())[0][0]
                y = list(dicts.values())[0][1]
                z = list(dicts.values())[0][2]
                radius = list(dicts.values())[3]

                if radius > 1.0:

                    feats.append(Feature(InteractionKind.ACCEPTOR, x, y, z, 1.0))

    
                else:

                    feats.append(Feature(InteractionKind.ACCEPTOR, x, y, z, radius))


                
            elif dicts[key] == "DONOR":

                x = list(dicts.values())[0][0]
                y = list(dicts.values())[0][1]
                z = list(dicts.values())[0][2]
                radius = list(dicts.values())[3]

                
                if radius > 1.0:

                    feats.append(Feature(InteractionKind.DONOR, x, y, z, 1.0))

            
                else:

                    feats.append(Feature(InteractionKind.DONOR, x, y, z, radius))



            elif dicts[key] == "HYDROFOBIC":


                x = list(dicts.values())[0][0]
                y = list(dicts.values())[0][1]
                z = list(dicts.values())[0][2]
                radius = list(dicts.values())[3]


                if radius > 1.0:

                    feats.append(Feature(InteractionKind.HYDROPHOBIC, x, y, z, 1.0))

            
                else:

                    feats.append(Feature(InteractionKind.HYDROPHOBIC, x, y, z, radius))


    pharmacophore_writer.write(feats, caminho_saida)




def create_pseudoatoms(caminho_ftmap: str, 
                            k_acceptor: int, 
                            k_donor: int, 
                            k_apolar:int, 
                            level:int,
                            saida:str):


    centroids = get_centroids_for_pharmacophore(caminho_ftmap, k_acceptor, k_donor, k_apolar, level)
    
    
    for i, dicts in enumerate(centroids):
        

        if dicts["TIPO"] == "ACCEPTOR":
                            
            x = list(dicts.values())[0][0]
            y = list(dicts.values())[0][1]
            z = list(dicts.values())[0][2]
            radius = list(dicts.values())[3]

            if radius > 1.0:

                cmd.pseudoatom(f"acc_{i}", pos=[x, y, z], elem= "O", chain= "Q", vdw= 1.0)

            
            else:

                cmd.pseudoatom(f"acc_{i}", pos=[x, y, z], elem= "O", chain= "Q", vdw= radius)


            cmd.color("red", f"acc_{i}")

        elif dicts["TIPO"] == "DONOR":

            x = list(dicts.values())[0][0]
            y = list(dicts.values())[0][1]
            z = list(dicts.values())[0][2]
            radius = list(dicts.values())[3]

            if radius > 1.0:

                cmd.pseudoatom(f"don_{i}", pos=[x, y, z], elem= "N", chain= "Q", vdw= 1.0)


            else:

                cmd.pseudoatom(f"don_{i}", pos=[x, y, z], elem= "N", chain= "Q", vdw= radius)


            
            cmd.color("blue", f"don_{i}")

        
        elif dicts["TIPO"] == "HYDROFOBIC":

            x = list(dicts.values())[0][0]
            y = list(dicts.values())[0][1]
            z = list(dicts.values())[0][2]
            radius = list(dicts.values())[3]

            if radius > 1.0:

                cmd.pseudoatom(f"apo_{i}", pos=[x, y, z], elem= "C", chain= "Q", vdw= 1.0)

        
            else:

                cmd.pseudoatom(f"apo_{i}", pos=[x, y, z], elem= "C", chain= "Q", vdw= radius)


            
            cmd.color("yellow", f"apo_{i}")



    cmd.show("spheres", "apo* don* acc*")
    cmd.save(f"{saida}/3eml_features_Level.pse", format= "pse")






def write_query_to_sybyl(caminho_ftmap: str,
                            caminho_saida:str,
                            k_acceptor: int, 
                            k_donor: int, 
                            k_apolar:int, 
                            level:int):
    
    centroids = get_centroids_for_pharmacophore(caminho_ftmap, k_acceptor, k_donor, k_apolar, level)
    
    

    coords_donor= []
    coords_acceptor = []
    coords_apolar = []

    sum_lines = []


    for dicts in centroids[0:3]:
        sum_lines.append(dicts)
        for key, values in dicts.items():

            if dicts[key] == "ACCEPTOR":
                
                x = list(dicts.values())[0][0]
                y = list(dicts.values())[0][1]
                z = list(dicts.values())[0][2]
                radius = list(dicts.values())[3]

                coords_acceptor.append((x,y,z,radius))

            elif dicts[key] == "DONOR":

                x = list(dicts.values())[0][0]
                y = list(dicts.values())[0][1]
                z = list(dicts.values())[0][2]
                radius = list(dicts.values())[3]

                coords_donor.append((x,y,z,radius))

            elif dicts[key] == "HYDROFOBIC":

                x = list(dicts.values())[0][0]
                y = list(dicts.values())[0][1]
                z = list(dicts.values())[0][2]
                radius = list(dicts.values())[3]

                coords_apolar.append((x,y,z, radius))

    
    with open(caminhos[3] + "1E66_query_17_3_Level_C.mol2", "w") as query:
        
        
        query.write(f"""
@<TRIPOS>MOLECULE
Complex_query
    0     0     0    {(len(sum_lines)*2) + 2}     0
SMALL
No_CHARGES


@<TRIPOS>ATOM2
@<TRIPOS>NORMAL 
@<TRIPOS>U_FEAT """)
        

        for i, (x, y, z, radius) in enumerate(coords_donor):  #get two first features ordenate by 


            if radius > 1.0:
            
                query.write(f""" 
4 13 AD_{i} DONOR_ATOM {x} {y} {z} 0 0 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 14 
5 14 SPAT_AD_{i} {1.0} {x} {y} {z} 0 1 AD_{i} BLUE""")


            else:

                query.write(f""" 
4 13 AD_{i} DONOR_ATOM {x} {y} {z} 0 0 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 14 
5 14 SPAT_AD_{i} {radius} {x} {y} {z} 0 1 AD_{i} BLUE""")


        for i, (x, y, z, radius) in enumerate(coords_apolar): #get two first features ordenate by DC

            if radius > 1.0:

                query.write(f""" 
4 13 HY_{i} HYDROPHOBIC {x} {y} {z} 0 0 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 14 
5 14 SPAT_HY_{i} {1.0} {x} {y} {z} 0 1 HY_{i} YELLOW""")

            else:

                query.write(f""" 
4 13 HY_{i} HYDROPHOBIC {x} {y} {z} 0 0 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 14 
5 14 SPAT_HY_{i} {radius} {x} {y} {z} 0 1 HY_{i} YELLOW""")


        for i, (x, y, z, radius) in enumerate(coords_acceptor): #get two first features ordenate by DC

            if radius > 1.0:

                query.write(f""" 
4 13 AA_{i} ACCEPTOR_ATOM {x} {y} {z} 0 0 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 14 
5 14 SPAT_AA_{i} {1.0} {x} {y} {z} 0 1 AA_{i} RED""")
           

            else:

                query.write(f""" 
4 13 AA_{i} ACCEPTOR_ATOM {x} {y} {z} 0 0 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 14 
5 14 SPAT_AA_{i} {radius} {x} {y} {z} 0 1 AA_{i} RED""")


        query.write("""
@<TRIPOS>RENDERING_ATTRS
ANTIALIASED_LINES 
*""")



caminhos= [ 
        "C:/Users/55749/Documents/3eml/3eml_box_v2.pdb", 
        "C:/Users/55749/Documents/3eml/hotspots_active_site.pse", 
        "C:/Users/55749/Documents/3eml/",
        "C:/Users/55749/Documents/3eml/resultados_novos/"
        ]



#write_query_to_sybyl(caminhos[0], caminhos[3], 3, 3, 4, 17)

#Xa, level_a = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], "acceptor", 17)
#Xd, level_d = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], "donor",  17)
#Xap, level_c = get_features_from_ftmap_and_fragmap(caminhos[0], caminhos[1], caminhos[2], "apolar", 17)

#metrics_clusters(Xa, caminhos[2], "acceptor", 17)
#metrics_clusters(Xd, caminhos[2], "donor", 17)
#metrics_clusters(Xap, caminhos[2], "apolar", 17)

#build_pharmacohore(caminhos[0], caminhos[3], 2, 2, 2, 17)
#write_query_to_sybyl(caminhos[0], caminhos[3], 5, 3, 4, 17)

#create_pseudoatoms(caminhos[0], 5, 2, 2, 17, caminhos[2])

#get_active_site("C:/Users/55749/Documents/3eml/3eml_box_v2.pdb", "C:/Users/55749/Documents/3eml/active_site.pdb")

create_pseudoatoms(caminhos[0], 5, 3, 2, 17, caminhos[2])