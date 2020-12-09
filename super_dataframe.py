# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:02:01 2020

Création d'un super DataFrame contenant toutes les interventions de
parlementaires.
Également création d'un DataFrame similaire mais uniquement avec les groupes
SOC, LFI, et LR pour entraîner le modèle sur des patis gauche/droite.

@author: Jérémie Stym-Popper
"""

import pandas as pd

LREM_df = pd.read_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\LREM2_inter.csv")
LFI_df = pd.read_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\LFI2_inter.csv")
LR_df = pd.read_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\LR2_inter.csv")
SOC_df = pd.read_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\SOC2_inter.csv")
AE_df = pd.read_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\AE2_inter.csv")
UDI_df = pd.read_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\UDI2_inter.csv")
LT_df = pd.read_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\LT2_inter.csv")
NG_df = pd.read_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\NG2_inter.csv")
GDR_df = pd.read_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\GDR2_inter.csv")
NI_df = pd.read_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\NI2_inter.csv")
UAI_df = pd.read_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\UAI2_inter.csv")
MODEM_df = pd.read_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\MODEM2_inter.csv")
gd_df = pd.read_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\gd_inter.csv")
all_df = pd.read_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\all_inter.csv")


"""
LFI_df = LFI_df.drop('Unnamed: 0', axis=1)
LREM_df = LREM_df.drop('Unnamed: 0', axis=1)
LR_df = LR_df.drop('Unnamed: 0', axis=1)
LT_df = LT_df.drop('Unnamed: 0', axis=1)
AE_df = AE_df.drop('Unnamed: 0', axis=1)
UDI_df = UDI_df.drop('Unnamed: 0', axis=1)
UAI_df = UAI_df.drop('Unnamed: 0', axis=1)
SOC_df = SOC_df.drop('Unnamed: 0', axis=1)
NG_df = NG_df.drop('Unnamed: 0', axis=1)
MODEM_df = MODEM_df.drop('Unnamed: 0', axis=1)
NI_df = NI_df.drop('Unnamed: 0', axis=1)
GDR_df = GDR_df.drop('Unnamed: 0', axis=1)
"""



#    LFI_df.to_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\LFI2_inter.csv", index=False)
 #   LREM_df.to_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\LREM2_inter.csv", index=False)
  #  MODEM_df.to_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\MODEM2_inter.csv", index=False)
   # UAI_df.to_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\UAI2_inter.csv", index=False)
    #NI_df.to_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\NI2_inter.csv", index=False)
    #AE_df.to_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\AE2_inter.csv", index=False)
    #GDR_df.to_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\GDR2_inter.csv", index=False)
    #NG_df.to_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\NG2_inter.csv", index=False)
    #UDI_df.to_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\UDI2_inter.csv", index=False)
    #LT_df.to_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\LT2_inter.csv", index=False)
    #LR_df.to_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\LR2_inter.csv", index=False)
   # SOC_df.to_csv(r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\SOC2_inter.csv", index=False)


super_dep_df = pd.concat([LFI_df, LREM_df, LR_df, LT_df, AE_df,
                          UDI_df, UAI_df, SOC_df, NG_df, MODEM_df,
                          NI_df, GDR_df])
super_dep_df = super_dep_df.reset_index()
super_dep_df = super_dep_df.drop('index', axis=1)

path_super = r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\all_inter.csv"
super_dep_df.to_csv(path_super)

gauche_droite_df = pd.concat([LFI_df, LR_df, SOC_df])
gauche_droite_df = gauche_droite_df.reset_index()
gauche_droite_df = gauche_droite_df.drop('index', axis=1)

path_gauche_droite = r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\new_repo_git\depythons\Stock_csv\gd_inter.csv"
gauche_droite_df.to_csv(path_gauche_droite)