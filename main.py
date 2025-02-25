import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from mlxtend.frequent_patterns import apriori, association_rules

file_path = "Groceries_dataset.csv"
groceries_df = pd.read_csv(file_path)

groceries_df['Date'] = pd.to_datetime(groceries_df['Date'], format='%d-%m-%Y')
groceries_df['itemDescription'] = groceries_df['itemDescription'].str.strip().str.lower()
groceries_df['Transaction_ID'] = groceries_df.groupby(['Member_number', 'Date']).ngroup()

basket = groceries_df.groupby(['Transaction_ID', 'itemDescription'])['itemDescription']\
                     .count()\
                     .unstack()\
                     .fillna(0)\
                     .astype(bool)

frequent_itemsets = apriori(basket, min_support=0.003, use_colnames=True)
frequent_itemsets['itemset_length'] = frequent_itemsets['itemsets'].apply(len)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.05)

# Формування текстового виводу
console_output = []
console_output.append("Очищення даних: Пропущені значення не знайдено.")
console_output.append("\nЧасті набори товарів:")
console_output.append(frequent_itemsets.head().to_string())
console_output.append("\nПравила асоціацій:")
console_output.append(rules.head().to_string())
console_output.append("\nГенерація правил асоціацій:")
console_output.append("Генерація корисних правил на основі виявлених наборів елементів, що часто зустрічаються.")
console_output.append("Правила асоціацій, які не досягають порогу в 1, відсікаються. Вище значення підйому означає, що правило є сильнішим/важливішим.")
console_output.append("Правила відсортовані в порядку спадання за значеннями достовірності та підйому.")
console_output.append("Чим більші значення довіри та підйому, тим сильніше правило.\n")
console_output.append(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by=['confidence', 'lift'], ascending=False).head().to_string())

app = dash.Dash(__name__)
app.title = "Аналіз ринкових кошиків"

item_frequencies = basket.sum().sort_values(ascending=False)

fig_top_items = px.bar(
    item_frequencies.head(10),
    x=item_frequencies.head(10).values,
    y=item_frequencies.head(10).index,
    labels={"x": "Частота", "y": "Товари"},
    title="Топ-10 найбільш популярних товарів",
    color=item_frequencies.head(10).values,
    color_continuous_scale="viridis"
)
fig_top_items.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))

fig_lift_confidence = px.scatter(
    rules,
    x="confidence",
    y="lift",
    title="Lift vs Confidence",
    labels={"confidence": "Довіра", "lift": "Підйом"},
    color="lift",
    size="support",
    color_continuous_scale="plasma"
)
fig_lift_confidence.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))

fig_itemset_length = px.histogram(
    frequent_itemsets,
    x="itemset_length",
    title="Розподіл довжини наборів товарів",
    labels={"itemset_length": "Довжина набору товарів"},
    color_discrete_sequence=["#EF553B"]
)
fig_itemset_length.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))

# Візуалізація зв'язків між товарами
G = nx.DiGraph()
for _, row in rules.iterrows():
    G.add_edge(str(row['antecedents']), str(row['consequents']), weight=row['lift'])

pos = nx.spring_layout(G)
edges = G.edges(data=True)
edge_x = []
edge_y = []
edge_text = []
for edge in edges:
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_text.append(f"Lift: {edge[2]['weight']:.2f}")

fig_network = go.Figure()
fig_network.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='white'), mode='lines', text=edge_text, hoverinfo='text'))
node_x = []
node_y = []
nodes_text = []
for node in pos:
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    nodes_text.append(node)
fig_network.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=nodes_text, textposition='top center', marker=dict(size=10, color='lightblue')))
fig_network.update_layout(title="Граф асоціацій між товарами", template="plotly_dark")

app.layout = html.Div([
    html.H1("Панель аналізу ринкових кошиків", style={'textAlign': 'center', 'color': 'white'}),
    html.Div([
        html.Div([
            dcc.Graph(id='top-items-graph', figure=fig_top_items)
        ], className="six columns"),
        html.Div([
            dcc.Graph(id='lift-confidence-graph', figure=fig_lift_confidence)
        ], className="six columns"),
    ], className="row"),
    html.Div([
        dcc.Graph(id='itemset-length-graph', figure=fig_itemset_length)
    ]),
    html.Div([
        dcc.Graph(id='network-graph', figure=fig_network)
    ]),
    html.Div([
        html.Pre("\n".join(console_output), style={"color": "white", "backgroundColor": "black", "padding": "10px", "whiteSpace": "pre-wrap"})
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)