<body>
  <h1>Iris Flower Classification using Logistic Regression</h1>

  <p>This project implements a machine learning classifier using <strong>Logistic Regression</strong> to classify the species of Iris flowers based on their physical attributes. It also includes a decision boundary visualization and model evaluation.</p>

  <h2>📁 Dataset</h2>
  <p>The model uses the classic <code>iris.csv</code> dataset, which contains the following features:</p>
  <ul>
    <li>Sepal Length (cm)</li>
    <li>Sepal Width (cm)</li>
    <li>Petal Length (cm)</li>
    <li>Petal Width (cm)</li>
    <li>Species (Target)</li>
  </ul>

  <h2>⚙️ Libraries Used</h2>
  <ul>
    <li>numpy</li>
    <li>pandas</li>
    <li>matplotlib</li>
    <li>seaborn</li>
    <li>scikit-learn</li>
  </ul>

  <h2>🚀 Workflow</h2>
  <ol>
    <li>Load and explore the dataset.</li>
    <li>Preprocess data and split it into training and testing sets.</li>
    <li>Train a <strong>Logistic Regression</strong> model.</li>
    <li>Evaluate the model using <em>accuracy score</em>, <em>confusion matrix</em>, and <em>classification report</em>.</li>
    <li>Visualize decision boundaries based on petal length and width.</li>
    <li>Perform sample prediction on new data.</li>
  </ol>

  <h2>📊 Model Evaluation</h2>
  <ul>
    <li><strong>Accuracy</strong> - Computed on test set.</li>
    <li><strong>Confusion Matrix</strong> - Visualized using seaborn heatmap.</li>
    <li><strong>Classification Report</strong> - Includes precision, recall, f1-score.</li>
  </ul>

  <h2>🌸 Sample Prediction</h2>
  <p>The model can predict the species of an Iris flower using input like:</p>
  <pre><code>classifier.predict([[5.4, 2.6, 4.1, 1.3]])</code></pre>

  <h2>📷 Visualizations</h2>
  <ul>
    <li>Confusion matrix heatmap for classification results.</li>
    <li>Decision boundary plot for 2D petal features.</li>
  </ul>

  <h2>📌 Note</h2>
  <p>To avoid errors, make sure you <code>call</code> methods using parentheses <code>()</code> and not brackets <code>[]</code>.</p>

  <h2>🧠 Author</h2>
  <p>Developed by Abhinav using scikit-learn and Python.</p>
</body>
</html>
