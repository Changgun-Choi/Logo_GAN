import pandas as pd
import numpy as np
import numpy.ma as ma
import igraph
from pyvis.network import Network
import pickle
# defined run methods
graph = True
clustering = False
visualization = False

# Script methods for vectorization


def pairwiseCombination(labelsArray):
    # Exclude element in Mask array to prevent processing of nan values
    # labelsArray = labelsArray.compressed()
    return np.array(np.meshgrid(labelsArray, labelsArray)).T.reshape(-1, 2)


def nodeNameExtraction(listIdx):
    return [graph.vs[idx].attributes().get('name') for idx in listIdx]


def clusterNameBetweeness(subGraph):
    graphBetweenness = subGraph.pagerank()
    maxBetweenness = max(graphBetweenness)
    maxBetweennessIdx = graphBetweenness.index(maxBetweenness)
    clusterName = subGraph.vs[maxBetweennessIdx].attributes().get('name')
    return clusterName


# read labels of images
data = pd.read_csv('FeatureCreation/LLD_GoogleLabels.csv')


# graph preparation
if graph is True:
    # node creation
    nodeName = np.unique(data['Labels'])
    nodeData = list(zip(nodeName, nodeName))
    nodes = pd.DataFrame.from_records(nodeData, columns=['id', 'label'])

    # edge creation
    # transform df values to list
    dfArray = data.groupby(by='Name')[
                           'Labels'].apply(list).to_numpy()
    # bring the unregularized list lengths to masked arrays
    # calculate maximum length of array
    npLength = max(map(len, dfArray))
    # build array with equal length filling with nan values
    labelsArray = np.array([xi+[np.nan]*(npLength-len(xi))
                           for xi in dfArray], dtype=object)
    # mask array where value is filled with invalid value
    maskedArray = ma.masked_where(pd.isna(labelsArray), labelsArray)
    # apply combination function on masked array
    edgesData = np.apply_along_axis(
        pairwiseCombination, 1, maskedArray).reshape(-1, 2)
    # create edge dataframe from edges array and drop none values
    edges = pd.DataFrame.from_records(
        edgesData, columns=['from', 'to']).dropna()
    edges = edges[~(edges['from'] == edges['to'])]
    # aggregate edges to get weight
    aggEdges = edges.groupby(['from', 'to']).size().reset_index(name='weight')
    # build up graph
    graph = igraph.Graph().DataFrame(edges, directed=True, vertices=nodes)

    # simplify graph based on fraction of edges
    # delete loops in edges
    graph = graph.simplify()

    # store information for ML-GCN network
    # extract count of each label per image
    labelCount = data.groupby(by='Labels').count()['Name'].values

    # calculate node similarity based on Jaccard coefficient
    graphJaccard = graph.similarity_jaccard()
    # transform from list to numpy array
    graphJaccard = np.array(graphJaccard)
    # extract edge list from graph
    edgeDataweight = graph.get_edge_dataframe()
    # assign weight based on node index by from and to
    edgeDataweight['weight'] = graphJaccard[edgeDataweight['source'],
                                            edgeDataweight['target']]
    # reduce edges that are under quantile
    fraction = 0.5
    reducedEdge = edgeDataweight[edgeDataweight.weight < edgeDataweight.groupby(
        'source').weight.transform('quantile', fraction)]
    # extract Index of to deleting edge index
    reducedEdgeIndex = reducedEdge.index.values
    # get edge weight based on opposite of to deleting edge index
    reducedEdgeWeight = edgeDataweight[~edgeDataweight.index.isin(
        reducedEdgeIndex)].weight.values
    # reduce edge from graph based on index
    graph.delete_edges(reducedEdgeIndex)
    # assign weight to edges
    graph.es['weight'] = reducedEdgeWeight
    # delete nodes from graph that have no edge present and are isolated
    # calculate list of degree (edge number) of all nodes and store in list
    nodeDegree = graph.degree()
    # get all index of nodeDegree where the degree is exactly 0
    emptyNodeIdx = [index for index,
                    degree in enumerate(nodeDegree) if degree == 0]
    # delete nodes of graph that are isolated based on list emptyNodeIdx
    graph.delete_vertices(emptyNodeIdx)
    # extract adjacency matrix for graph
    adjArray = graph.get_adjacency()
    adjArray = np.array(adjArray.data)
    # create dict consistiong of adjArray and numbers
    dictAdj = {'adj': adjArray, 'nums': labelCount}
    # write created dict to pickle file
    with open('FeatureCreation/logo_adj.pkl', 'wb') as f:
        pickle.dump(dictAdj, f)
    # summary information for graph
    print(f'Graph build finished. Summary: {graph.summary()}')

if clustering is True:
    # run clustering
    # cluster = graph.community_fastgreedy().as_clustering()
    cluster = graph.community_multilevel()
    subGraphCluster = cluster.subgraphs()
    nodeIdxCluster = list(cluster)
    nodeNameCluster = list(map(nodeNameExtraction, nodeIdxCluster))

    # determine cluster names
    clusterName = list(map(clusterNameBetweeness, subGraphCluster))
    clusterName
    clusterNameNodes = list(zip(clusterName, nodeNameCluster))
    clusterNameNodes
    clusterNameNodesData = [(clusterTuple[0], node)
                            for clusterTuple in clusterNameNodes
                            for node in clusterTuple[1]]
    clusterNameNodesData
    # build node-cluster dataframe
    nodeCluster = pd.DataFrame.from_records(
        clusterNameNodesData, columns=['cluster', 'node'])
    nodeCluster['cluster'].unique()
    nodeCluster['cluster'] = nodeCluster['cluster'].astype('category')
    nodeCluster['cluster_cat'] = nodeCluster['cluster'].cat.codes
    imageCluster = pd.merge(left=data, right=nodeCluster,
                            how='left', left_on='Labels', right_on='node')
    imageCluster = imageCluster[['Name', 'Labels', 'Scores', 'cluster']]
    imageCluster.to_csv('FeatureCreation/imageCluster.csv', index=False)
    # generate different image-label cluster csv
    # generate general cluster overview
    clusterDf = pd.DataFrame({'count': imageCluster.groupby(
        by=['Name', 'cluster']).size()}).reset_index()

    clusterDf['meanScore'] = imageCluster.groupby(
        by=['Name', 'cluster']).mean().values
    clusterDf['weightedScore'] = clusterDf['count'] * clusterDf['meanScore']

    # generate majority vote df
    majorityVote = clusterDf.iloc[clusterDf.groupby(
        'Name')['count'].agg(pd.Series.idxmax)][['Name', 'cluster']]
    majorityVote.to_csv('Model/Label/LLD_majorityVote.csv', index=False)
    # generate weighted majority vote
    weightedMajorityVote = clusterDf.iloc[clusterDf.groupby(
        'Name')['weightedScore'].agg(pd.Series.idxmax)][['Name', 'cluster']]
    weightedMajorityVote.to_csv(
        'Model/Label/LLD_weightedMajorityVote.csv', index=False)
    print(f'{len(nodeIdxCluster)} Graph clusters identified')

# visualization of network
if visualization is True:
    net = Network('1200px', '1200px')
    nodesVisId = list(nodeCluster['node'])
    nodesVisCluster = list(nodeCluster['cluster_cat'])
    net.add_nodes(nodesVisId, group=nodesVisCluster)
    print('nodes added')
    graphEdges = graph.es()
    visEdges = [edge.tuple for edge in graphEdges]
    visFromEdges = [edge[0] for edge in visEdges]
    visToEdges = [edge[1] for edge in visEdges]
    visFromEdges = nodeNameExtraction(visFromEdges)
    visToEdges = nodeNameExtraction(visToEdges)
    visEdges = list(zip(visFromEdges, visToEdges))
    net.add_edges(visEdges)
    print('Edges added')
    net.show_buttons(filter_=['physics'])
    net.write_html('FeatureCreation/ImageLabelsNetwork.html')
    print('Visualization finished')
