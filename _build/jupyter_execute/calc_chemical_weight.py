#!/usr/bin/env python
# coding: utf-8

# # Calculation of chemical weight<br/>
# Lei Lei<br/>
# A script uses the molmass package to calculate the weight of raw chemials required for synthesis.

# In[1]:


from molmass import Formula


# In[2]:


# Test the mass method of formula object
LSTO=Formula("La10Sr85Ti100O300", groups=None)
LSTO.mass/100


# In[3]:


# Test the composition method of formula object
TiO2=Formula("TiO2")
TiO2.composition(False)


# Input data, the information of chemicals and target compouds.

# In[4]:


raw_materials={
    "La": "La2O3",
    "Sr": "SrCO3",
    "Ti": "TiO2",
}

# Compounds and grams to synthesis
TargetCompounds={
    "La10Sr85Ti100O300": 5,
    "La20Sr70Ti100O300": 5,
    "La30Sr55Ti100O300": 5,
    "La40Sr40Ti100O300": 5,
}


# Calculate the number of atoms and molar mass of the chemicals.

# In[5]:


# Compositions & number
for element, chemical in raw_materials.items():
    chemical_formula=Formula(chemical)
    compositions=chemical_formula.composition(False)
    for composition in compositions:
        if composition[0]==element:
            multiplicity=composition[1]
            globals()[chemical]=[multiplicity, chemical_formula.mass]
    print(chemical +":", globals()[chemical])


# Calculate the moles of elements, store the results into a dict.

# In[6]:


for compound, weight in TargetCompounds.items():
    LSTO=Formula(compound)
    globals()[compound+"_moles"], globals()[compound+"_composition"]=weight/LSTO.mass, LSTO.composition(False)
    # print(globals()[compound+"_moles"], "\n", globals()[compound+"_composition"])
    globals()[compound] = {}
    for element in globals()[compound+"_composition"]:
        if any(element[0] in key for key in raw_materials):
            # The moles of elements
            globals()[compound][element[0]] = globals()[compound+"_moles"]*element[1]
    print(globals()[compound])


# Define a function to calculate the weight of raw materials:
# $$
# m(\mathrm{chemical})=\frac{n(\mathrm{element}) \times M(\mathrm{chemical})}{N}
# $$
# where $M(\mathrm{chemical})$ and $N$ are the molar mass and multiplicity (<i>i.e.</i> the number of atoms in the chemical formula) of the element. 

# In[7]:


def calc_weight(n, mm, multiple):
    return n * mm / multiple


# Calculate the weight of chemicals to use by the `calc_weight` function.

# In[8]:


for compound, weight in TargetCompounds.items():
    globals()[compound+"_chemicals"]={}
    for element, chemical in raw_materials.items():
        globals()[compound+"_chemicals"][chemical] = calc_weight (globals()[compound][element], globals()[chemical][1], globals()[chemical][0])
    print(globals()[compound+"_chemicals"])


# In[9]:


# Write a markdown file as record.
from time import gmtime, strftime
Title="# Calculation of chemical weights\n"
author="Lei Lei\n"
time = strftime("%Y-%m-%d %H:%M:%S\n", gmtime())
description="Data generated by the code in the jupyter notebook `calc_chemical_weight.ipynb`."
filename="AD-compositions.md"

with open (filename, 'w') as file:
    file.write(Title)
    file.write(author)
    file.write(time)
    file.write(description)
    for compound, weight in TargetCompounds.items():
        file.write("## " + compound + "\n")
        for key, value in globals()[compound+"_chemicals"].items():
            file.write("{0}: {1:.4f} g\n".format(key, value))
file.close

