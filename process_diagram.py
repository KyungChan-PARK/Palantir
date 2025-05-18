from graphviz import Digraph

dot = Digraph(comment='Data Quality Validation Pipeline')

dot.node('A', 'Start')
dot.node('B', 'Load Data')
dot.node('C', 'Validate Data')
dot.node('D', 'Generate Report')
dot.node('E', 'Send Notification')
dot.node('F', 'End')

# Define edges

dot.edges(['AB', 'BC', 'CD', 'DE', 'EF'])

# Save and render the diagram
dot.render('data_quality_validation_pipeline', format='png', cleanup=True) 