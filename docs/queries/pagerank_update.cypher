// pagerank_update.cypher
// Recompute PageRank on Topic prerequisite graph and write score to property `pr_score`
CALL gds.graph.project(
  'topicGraph',
  'Topic',
  {
    PREREQ_OF: {orientation:'NATURAL'}
  }
) YIELD graphName;

CALL gds.pageRank.write('topicGraph', {
  writeRelationshipType: 'PREREQ_PR',
  writeProperty: 'pr_score'
});

CALL gds.graph.drop('topicGraph'); 