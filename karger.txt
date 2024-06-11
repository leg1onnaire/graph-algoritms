import networkx as nx
import matplotlib.pyplot as plt
import random
import time

def random_node(G):
    node1 = random.choice(list(G.nodes()))
    node2 = random.choice(list(G.nodes()))
    if node1 != node2 and G.has_edge(node1,node2):
        return node1, node2
    elif node1 == node2 or not G.has_edge(node1,node2):
        node1i = random.choice(list(G.nodes()))
        node2i = random.choice(list(G.nodes()))
        while not G.has_edge(node1i, node2i) or node1i == node2i:
            node1i = random.choice(list(G.nodes()))
            node2i = random.choice(list(G.nodes()))
    return node1i, node2i

def Contract(G):
    Gi = G.copy()
    if G.number_of_nodes() > 2:
        while Gi.number_of_nodes() > 2:
            node1, node2 = random_node(Gi)
            #print(node1, '----', node2)
            Gi = nx.contracted_nodes(Gi,node1, node2,self_loops=False,copy=True)
            Gi = nx.relabel_nodes(Gi, {node1 : node1+','+node2})
            grph_list = list(Gi.degree())
            #print(grph_list)
            #pos2 = nx.spring_layout(Gi)
            #nx.draw(Gi, pos2, with_labels=True, node_size=500, node_color='Blue', font_weight='bold')
            #plt.show()
        return grph_list
    elif G.number_of_nodes() == 2:
        return list(G.degree)

def Smallest_Cut(G):
    bound = G.number_of_nodes() * G.number_of_nodes()
    iteration_dict = {}
    copy_dict = {}
    smallest_key = 1

    start_t = time.time()
    for i in range(1, bound + 1):
        list_func = Contract(G)
        iteration_dict[i] = (list_func[0][1], list_func[0][0], list_func[1][0])
    end_t = time.time()

    print(iteration_dict)
    print("Time elapsed : ", end_t - start_t, "seconds")

    for i in range(1, bound + 1):
        copy_dict[i] = iteration_dict[i][0]
    for i in range(1, bound + 1):
        if copy_dict[i] < copy_dict[smallest_key]:
            smallest_key = i
        else:
            continue
    print("smallest cut is : ", iteration_dict[smallest_key], "-------", smallest_key)

    lst1 = list((iteration_dict[smallest_key][1]).split(','))
    lst2 = list((iteration_dict[smallest_key][2]).split(','))
    print(lst1)
    print(lst2)
    grp1 = nx.subgraph(G, lst1)
    grp2 = nx.subgraph(G, lst2)
    return grp1, grp2

    #return list(iteration_dict[smallest_key])

G = nx.MultiGraph()
#nodes = ['1','2','3','4','5','6']
nodes = ['1','2','3','4','5','6','7','8','9','10']
#nodes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103']
'''
nodes = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
    '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
    '31', '32', '33', '34'
]
'''
'''
nodes = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
    '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
    '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58',
    '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75'
]
'''
G.add_nodes_from(nodes)
edges = [('1','2'),('1','3'),('1','4'),('1','5'),('2','3'),('2','4'),('2','5'),('3','4'),('3','5'),('4','5'),('6','7'),('6','8'),('6','9'),('6','10'),('7','8'),('7','9'),('7','10'),('8','9'),('8','10'),('9','10'),('3','7'),('4','6'),('5','10')]
#edges = [('1','2'), ('1','3'), ('1','4'), ('3','4'), ('1','5'), ('2','5'), ('4','5'),('6','2')]
'''
edges = [
    ('1', '2'), ('1', '3'), ('1', '4'), ('1', '5'), ('2', '3'), ('2', '6'), ('2', '7'), ('3', '4'), ('3', '8'), ('4', '9'),
    ('4', '10'), ('5', '11'), ('5', '12'), ('5', '13'), ('6', '14'), ('6', '15'), ('7', '16'), ('7', '17'), ('8', '18'), ('8', '19'),
    ('9', '20'), ('9', '21'), ('10', '22'), ('10', '23'), ('11', '24'), ('11', '25'), ('12', '26'), ('12', '27'), ('13', '28'), ('13', '29'),
    ('14', '30'), ('14', '31'), ('15', '32'), ('15', '33'), ('16', '34'), ('16', '35'), ('17', '36'), ('17', '37'), ('18', '38'), ('18', '39'),
    ('19', '40'), ('19', '41'), ('20', '42'), ('20', '43'), ('21', '44'), ('21', '45'), ('22', '46'), ('22', '47'), ('23', '48'), ('23', '49'),
    ('24', '50'), ('24', '51'), ('25', '52'), ('25', '53'), ('26', '54'), ('26', '55'), ('27', '56'), ('27', '57'), ('28', '58'), ('28', '59'),
    ('29', '60'), ('29', '61'), ('30', '62'), ('30', '63'), ('31', '64'), ('31', '65'), ('32', '66'), ('32', '67'), ('33', '68'), ('33', '69'),
    ('34', '70'), ('34', '71'), ('35', '72'), ('35', '73'), ('36', '74'), ('36', '75'), ('37', '76'), ('37', '77'), ('38', '78'), ('38', '79'),
    ('39', '80'), ('39', '81'), ('40', '82'), ('40', '83'), ('41', '84'), ('41', '85'), ('42', '86'), ('42', '87'), ('43', '88'), ('43', '89'),
    ('44', '90'), ('44', '91'), ('45', '92'), ('45', '93'), ('46', '94'), ('46', '95'), ('47', '96'), ('47', '97'), ('48', '98'), ('48', '99'),
    ('49', '100'), ('49', '101'), ('50', '102'), ('50', '103')
]

edges = [
    ('2', '1'),
    ('3', '1'), ('3', '2'),
    ('4', '1'), ('4', '2'), ('4', '3'),
    ('5', '1'),
    ('6', '1'),
    ('7', '1'), ('7', '5'), ('7', '6'),
    ('8', '1'), ('8', '2'), ('8', '3'), ('8', '4'),
    ('9', '1'), ('9', '3'),
    ('10', '3'),
    ('11', '1'), ('11', '5'), ('11', '6'),
    ('12', '1'),
    ('13', '1'), ('13', '4'),
    ('14', '1'), ('14', '2'), ('14', '3'), ('14', '4'),
    ('17', '6'), ('17', '7'),
    ('18', '1'), ('18', '2'),
    ('20', '1'), ('20', '2'),
    ('22', '1'), ('22', '2'),
    ('26', '24'), ('26', '25'),
    ('28', '3'), ('28', '24'), ('28', '25'),
    ('29', '3'),
    ('30', '24'), ('30', '27'),
    ('31', '2'), ('31', '9'),
    ('32', '1'), ('32', '25'), ('32', '26'), ('32', '29'),
    ('33', '3'), ('33', '9'), ('33', '15'), ('33', '16'), ('33', '19'), ('33', '21'), ('33', '23'), ('33', '24'), ('33', '30'), ('33', '31'), ('33', '32'),
    ('34', '9'), ('34', '10'), ('34', '14'), ('34', '15'), ('34', '16'), ('34', '19'), ('34', '20'), ('34', '21'), ('34', '23'), ('34', '24'), ('34', '27'), ('34', '28'), ('34', '29'), ('34', '30'), ('34', '31'), ('34', '32'), ('34', '33')
]
'''
'''
edges = [
    ('48', '62'), ('62', '33'), ('33', '7'), ('7', '58'), ('58', '69'), ('69', '45'), ('45', '19'), ('19', '13'), ('13', '20'), ('20', '44'),
    ('44', '43'), ('43', '54'), ('54', '66'), ('66', '21'), ('21', '73'), ('73', '68'), ('68', '28'), ('28', '35'), ('35', '26'), ('26', '27'),
    ('27', '46'), ('46', '40'), ('40', '5'), ('5', '47'), ('47', '15'), ('15', '41'), ('41', '12'), ('12', '75'), ('75', '4'), ('4', '53'),
    ('53', '37'), ('37', '31'), ('31', '50'), ('50', '10'), ('10', '56'), ('56', '63'), ('63', '6'), ('6', '24'), ('24', '38'), ('38', '71'),
    ('71', '64'), ('64', '32'), ('32', '34'), ('34', '36'), ('36', '51'), ('51', '1'), ('1', '49'), ('49', '59'), ('59', '29'), ('29', '39'),
    ('39', '16'), ('16', '60'), ('60', '67'), ('67', '22'), ('22', '9'), ('9', '3'), ('3', '57'), ('57', '70'), ('70', '55'), ('55', '30'),
    ('30', '18'), ('18', '72'), ('72', '52'), ('52', '11'), ('11', '23'), ('23', '14'), ('14', '8'), ('8', '74'), ('74', '65'), ('65', '42'),
    ('42', '2'), ('2', '61'), ('61', '17'), ('17', '25'), ('25', '48'), ('18', '37'),
    ('4', '68'),
    ('59', '71'),
    ('22', '62'),
    ('13', '47'),
    ('28', '55'),
    ('30', '73'),
    ('6', '36'),
    ('2', '67'),
    ('15', '74'),
    ('12', '48'),
    ('32', '60'),
    ('9', '51'),
    ('21', '41'),
    ('1', '69'),
    ('46', '75'),
    ('16', '52'),
    ('19', '65'),
    ('24', '61'),
    ('7', '40'),
]
'''


G.add_edges_from(edges)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=500, node_color='Blue', font_weight='bold')
plt.show()
#print(G.nodes())
print("Start graph degree : ", G.degree())

groups_list = []
num_of_groups = 1
k=2
group1, group2 = Smallest_Cut(G)

#pos1 = nx.spring_layout(group1)
#nx.draw(group1, pos1, with_labels=True, node_size=500, node_color='Red', font_weight='bold')
#plt.show()
print(group1.number_of_nodes())
#pos2 = nx.spring_layout(group2)
#nx.draw(group2, pos2, with_labels=True, node_size=500, node_color='Red', font_weight='bold')
#plt.show()
print(group2.number_of_nodes())


og_ed = edges
num_node_og = 6
subsets = []
subsets.append(list(group1.nodes()))
subsets.append(list(group2.nodes()))
print(subsets)

def graph_plot(original_edges, num_nodes_original, subsets):
    # Create graphs for original and partitioned edges
    G_original = nx.Graph()
    G_partitioned = nx.Graph()

    # Add original edges to G_original
    G_original.add_edges_from(original_edges)

    # Add partitioned edges to G_partitioned
    G_partitioned.add_edges_from(original_edges)  # Retain original edges for the partitioned graph

    # Get positions for nodes using spring layout
    pos_original = nx.spring_layout(G_original)
    pos_partitioned = nx.spring_layout(G_partitioned)

    # Plot the graphs side by side
    plt.figure(figsize=(12, 5))

    # Plot the original graph
    plt.subplot(1, 2, 1)
    plt.title('Original Graph')
    nx.draw(G_original, pos_original, with_labels=True, node_size=500, node_color='skyblue', font_weight='bold',
            font_color='black')

    # Generate distinct colors using a color map
    num_subsets = len(subsets)
    color_map = plt.cm.get_cmap('tab10', num_subsets)  # Use a colormap to generate distinct colors

    # Plot the partitioned graph with different subsets having different colors
    plt.subplot(1, 2, 2)
    plt.title('Partitioned Graph')
    for i, subset in enumerate(subsets):
        color = plt.cm.Set1(i / len(subsets)) # Get a color from the colormap
        nx.draw_networkx_nodes(G_partitioned, pos_partitioned, nodelist=subset, node_color=color, node_size=500)

    nx.draw_networkx_edges(G_partitioned, pos_partitioned, edgelist=G_partitioned.edges(), width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G_partitioned, pos_partitioned, font_color='black', font_weight='bold')

    plt.tight_layout()
    plt.show()

graph_plot(og_ed,num_node_og,subsets)