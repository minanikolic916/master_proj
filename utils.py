
class FormatedNode:
    def __init__(self, id, file_name, text, score, ret_score):
        self.id = id
        self.file_name = file_name
        self.text = text
        self.score = score
        self.ret_score = ret_score
        
def format_nodes(nodes):
    formated_nodes = []
    for node in nodes:
        formated_node = FormatedNode(
            id = node.id_,
            file_name = node.node.metadata["file_name"], 
            text = node.node.get_content(),
            score = node.score, 
            ret_score = node.node.metadata["retrieval_score"]
        )
        formated_nodes.append(formated_node)
    return formated_nodes

def get_context_node(formated_nodes):
    context = formated_nodes[0].text
    return context

      