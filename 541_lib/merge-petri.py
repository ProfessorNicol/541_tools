import pdb
import json
import copy
import sys

# validateFile tries to open the file whose path is given,
# and throws a terminal error if it cannot.  An open file
# is attempted to be loaded as a json file, and a terminal
# error is thrown on an error.
#  If no error is thrown the dictionary resulting from loading
# the json file is returned.

def validateFile(filename):
    try:
        rf = open(filename,"r")
    except Exception as e:
        print(f"Error opening file {filename} (e)")
        exit(1)

    try:
        jdict = json.load(rf)
    except json.JSONDecodeError as e:
        print(f"Invalid json format for {filename} (e)")
        exit(1)

    return jdict


# readMapping is passed the path to a json file containing a 
# description of the submodels and their common places.
# On successful execution it returns (1) a dictionary that is indexed
# by submodel identifier, and whose value is a a dictionary read
# in by a json load of the submodel description in the format 
# specified for Petri nets.  It also returns (2) a dictionary
# that described the common places, defined in the format
# of the merging Petri net configuration.

def readMapping(mapping_file):

    # get the merging configuration dictionary
    mrgDict = validateFile(mapping_file)

    # submodelDict will build up the (1) output
    submodelDict = {}

    # the merge configuration file 'submodel' value is a list of
    # codes that bind submodel name with file path to a json file
    # giving a Petri net description of that submodel

    for sm_code in mrgDict['submodels']:

        # the code separated submodel name from path by a comma
        if sm_code.find(',') == -1:
            raise Exception(f"no ',' in submodel code {sm_code}") 
        pieces = sm_code.split(',')
        if len(pieces) != 2:
            raise Exception(f"submodel code {sm_code} has too many ','") 
        stripped = [s.strip() for s in pieces]

        # assign the model and file path after cleanup
        model_name, file_path = stripped

        # read in the submodel's Petri net description
        submodelDict[model_name] = validateFile(file_path)

    # return the submodel name -> submodel dictionary assignment,
    # and the merge configuration's list of common place descriptions
    return submodelDict, mrgDict['mapping']

# createPlaceMap's input is a list of submodel names, and the 
# list of common place descriptions from the merge configuration file.
# The input is a list of submodel names, and the list of mapping dictionies from
# the merge configuration file (one per set of common places).
#   The method returns (1) a dictionary of dictionaries, the outside index
# being a submodel name, the referenced dictionary being a mapping
# that gives the merged name of a submodel place name (all places in
# all submodels that are declared to be common have this name).
#  createPlaceMap also returns a dictionary that gives, for every
# common place name, its initial marking (specified in the merge configuration
# file)

def createPlaceMap(submodels, mapping):

    # initialize output dictionaries
    placeMap = {}
    marking  = {}

    # for each submodel initialize the dictionary of submodel place name to common name
    for submodel_name in submodels:
        placeMap[submodel_name] = {}

    # mapping is a list of dictionaries describing the association of places in different
    # submodels
    for mapDict in mapping:

        # make sure the mapping dictionary has the attributes expected
        if not ('same' in mapDict and 'name' in mapDict and 'marking' in mapDict):
            raise Exception(f"mapping dictionary {mapDict} lacks expected attribute") 

        # the string associated with the 'same' keyword is a comma-separated list of places
        places = mapDict['same'].split(',')

        # the string associated with the 'name' keyword is the common name for all associated places
        commonName = mapDict['name']

        # move the initial marking out of the common place dictionary into one
        # dedicated to describing initial markings of shared places
        marking[commonName] = mapDict['marking']

        # every string representing a shared place is assumed to have the format of
        # a submodel name and ':' as prefix to the place name as it appears in the
        # submodel's Petri net description dictionary
        for smPlace in places:

            # make sure the expected format is given
            if smPlace.find(':') == -1:
                raise Exception(f"submodel place designator {smPlace} lacks ':'") 
            pieces = smPlace.split(':')
            if len(pieces) != 2:
                raise Exception(f"submodel place designator {smPlace} has too many ':'") 

            stripped = [s.strip() for s in pieces]

            # save the cleaned up model name and place name as given in the submodel's Petri net
            # description in variables model_name and place_name
            model_name, place_name = stripped

            # make sure we recognize the submodel name, remembering that the placeMap
            # dictionary is indexed by submodel name
            if model_name not in placeMap:
                raise Exception(f"submodel place designator {smPlace} reference to unrecognized submodel {model_name}") 

            # we can now save the mapping of local place name in the submodel Petri net dictionary
            # with the common name is shares with other submodels
            placeMap[model_name][place_name] = commonName

    # return the results
    return placeMap, marking

# createPetri takes a path to a merge configuration json file, and a
# path to write the merged Petri net description to.   It opens the
# merge configuration and creates a merged description as is specified,
# and writes out a json description of the merged Petri net

def createPetri(mapping_file, output_file):
   
    # submodelDict has model name as index and dictionary of submodel as value,
    # mappingLisg is the list of dictionaries in the 'mapping' list
    # from the merging input file, each of which describes one equivalence among places
    
    submodelDict, mappingList = readMapping(mapping_file)

    # submodel_place_dict is a dictionary which is indexed by the submodel name 
    # and creates as value for it a dictionary that is indexed by the place names 
    # in the submodel and gives the common place name that will replace it.  
    #  marking is a dictionary that is indexed by the common
    # place name and gives the marking for that place
    submodel_place_dict, marking = createPlaceMap(list(submodelDict.keys()), mappingList)   

    # to validate the placeMap we need to examine every submodel and check whether
    # the name given in the modelMap as place_name exists

    for submodel, smDict in submodelDict.items():
        for place_name in submodel_place_dict[submodel]:
            if not place_name in smDict['places']:
                raise Exception(f"place name {place_name} listed to be common not recognized in declared submodel {submodel}") 

    # merged_places will hold a list of the names of all places in the merged Petri net.
    # We initialize it with the names of the places that are shared by submodels    
    merged_places = copy.copy(list(marking.keys()))

    # we initialize the dictionary of initial markings for all places in the merged Petri net
    # by the now-known marking of the common places
    merged_marking = copy.copy(marking)

    # initialize the other two commons of the eventual output Petri net description
    merged_trans = []
    merged_rates   = {}

    # visit each submodel, using its Petri net description as a basis
    # for creating places and transitions in the merged Petri that are
    # copied---but renamed---instances of places and transitions in the submodel

    for submodel, smDict in submodelDict.items():

        # initialize dictionaries that will record how we have remapped
        # the place and transition names
        replace_place = {}
        replace_trans = {}

        # remember the marking of non-common places specified in the submodel Petri net description
        non_common_places = []

        # visit each place in the submodel, and note its remapped name in the merged
        # Petri net, the transformation depending on whether the places is common
        # between submodels or not

        for s in smDict['places']:
            # was s declared to be common?
            if s in submodel_place_dict[submodel]:
                # remap to the common place name, nestled in the submodel_place_dict dictionary
                replace_place[s] = submodel_place_dict[submodel][s]
            else:
                # not common, so prepend 'submodel:' to the place name
                # to differentiate it from other places in the merged model
                new_name = submodel+":"+s
                non_common_places.append(new_name)
                replace_place[s] = new_name

        # include the list of renamed non-common places in the list being built up for the
        # merged model
        merged_places.extend(non_common_places)

        # visit every submodel transition, copy it, and change the submodel
        # names of all input and output places to their representation in the merged Petri net
        for sub_trans in smDict['transitions']:
            new_trans = copy.copy(sub_trans)

            # prepend "submodel:" to the transition name
            new_trans['name'] = submodel+':'+sub_trans['name']

            # the replace_place dictionary tells how to remap the expression
            # of all places in this submodel
            new_trans['input_places']  = [ replace_place[s] for s in sub_trans['input_places']]
            new_trans['output_places'] = [ replace_place[s] for s in sub_trans['output_places']]

            # remember the transformed transition name to be used when
            # we merge transition rates  
            replace_trans[ sub_trans['name'] ] = new_trans['name']

            # include the transformed transition to the list of merged Petri net transitions
            merged_trans.append(new_trans)

        # include this submodel's markings of non-common places in the marking dictionary
        # for the merged Petri net 
        for name, marking_value in smDict['marking'].items():
            # is this a common place?  Remember that the keys of dictionary
            # submodel_place_dict[submodel] are submodel place names for 
            # places that are common to other submodels
            if name in submodel_place_dict[submodel]:
                continue

            # is non-common, so we can just copy the value of the marking
            # as given in the submodel
            merged_marking[ replace_place[name] ] = marking_value

        # replace_trans maps the name of every transition in the submodel to
        # its name in the merged model, so we can just copy the transition
        # rates as given in the submodel to the merged model
        for name, rate in smDict['transition_rates'].items():
            merged_rates[ replace_trans[name] ] = rate 

        # create a dictionary that will yield a json file in the
        # format expected for a Petri net description

        output_dict = { "name": "merged_topo", "net_type":"timed", \
                "places": merged_places,
                "transitions": merged_trans,
                "marking": merged_marking,
                "transition_rates": merged_rates
        }

        # write it out to the indicated file
        try: 
            with open(output_file, "w") as wf:
                json.dump(output_dict, wf, indent=4)
        except Exception as e:
            print(f"Error writing json output file {output_file} (e)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception(f"use {sys.argv[0]} mergeFile outputFile") 

    createPetri(sys.argv[1], sys.argv[2])


