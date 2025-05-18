/// <reference types="cypress" />

// Basic end-to-end test for the Dash app

describe('Knowledge Map App', () => {
  it('loads the main page', () => {
    cy.visit('http://localhost:8050');
    cy.contains('Knowledge Map');
    cy.get('#graph');
    cy.get('#pagerank-bar');
  });
});

