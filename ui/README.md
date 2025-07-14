# HSC Semantic Search - Web UI

A beautiful, interactive web interface for the Holistic Semantic Continuum (HSC) system, built with Streamlit.

## 🚀 Features

### 🏠 **Dashboard**
- System overview with key metrics
- Quick search functionality
- Recent activity tracking
- Real-time status monitoring

### 🔍 **Semantic Search**
- Interactive code search by meaning
- Real-time results with similarity scores
- Code preview with syntax highlighting
- Search result visualizations

### 🔄 **Code Transformation**
- Algorithm morphing via interpolation
- Interactive transformation paths
- Step-by-step transformation visualization
- Confidence score tracking

### 📊 **Analytics Dashboard**
- Comprehensive system statistics
- Category distribution analysis
- Similarity pattern analysis
- Code length distribution
- Performance metrics

### 🎨 **Interactive Visualizations**
- 2D/3D embedding space plots
- Interactive similarity matrices
- Transformation path animations
- Category distribution charts
- Real-time data exploration

### ⚡ **Performance Benchmarks**
- Automated benchmark execution
- Performance visualization
- Search time analysis
- System optimization insights

## 🛠️ Installation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the UI:**
   ```bash
   python run_ui.py
   ```
   
   Or directly with Streamlit:
   ```bash
   streamlit run ui/app.py
   ```

3. **Open Browser:**
   Navigate to `http://localhost:8501`

## 📱 UI Components

### Navigation
- **Sidebar**: Easy navigation between different sections
- **Responsive Design**: Works on desktop and mobile
- **Modern Styling**: Beautiful gradients and animations

### Interactive Elements
- **Real-time Search**: Instant results as you type
- **Clickable Visualizations**: Explore data interactively
- **Expandable Results**: Detailed code previews
- **Progress Indicators**: Loading states and feedback

### Data Visualization
- **Plotly Charts**: Interactive, zoomable, and exportable
- **3D Plots**: Immersive embedding space exploration
- **Heatmaps**: Similarity matrix analysis
- **Bar Charts**: Category and performance analysis

## 🎯 Usage Examples

### Semantic Search
1. Go to the **Search** page
2. Enter a query like `"def search(graph, start):"`
3. View real-time results with similarity scores
4. Click on results to see detailed code

### Code Transformation
1. Go to the **Transform** page
2. Select start and end algorithms
3. Choose number of transformation steps
4. Watch the transformation path visualization
5. Explore intermediate code steps

### Analytics
1. Go to the **Analytics** page
2. View system statistics and metrics
3. Analyze similarity patterns
4. Explore category distributions
5. Monitor performance trends

### Visualizations
1. Go to the **Visualizations** page
2. Explore 2D/3D embedding spaces
3. Analyze similarity matrices
4. Generate custom plots
5. Export visualizations

## 🔧 Configuration

### Customization
- Modify `ui/app.py` to add new features
- Update CSS in the `st.markdown` section
- Add new visualization functions
- Customize color schemes and layouts

### Performance
- Results are cached for faster loading
- Large datasets are processed efficiently
- Interactive plots use optimized rendering
- Search results are paginated

### Security
- Local development server only
- No external data transmission
- Secure file handling
- Input validation and sanitization

## 📊 Data Sources

The UI connects to the HSC semantic search engine and displays:
- Algorithm embeddings from TheAlgorithms/Python
- Semantic similarity scores
- Transformation paths
- Performance benchmarks
- System statistics

## 🎨 Design Philosophy

- **Modern**: Clean, professional interface
- **Interactive**: Real-time feedback and exploration
- **Informative**: Rich data visualization
- **Accessible**: Easy navigation and clear labels
- **Responsive**: Works on all screen sizes

## 🚀 Future Enhancements

- **User Authentication**: Multi-user support
- **Custom Datasets**: Upload your own code
- **Advanced Filters**: Category and similarity filtering
- **Export Features**: Save results and visualizations
- **API Integration**: REST API for external tools
- **Mobile App**: Native mobile interface

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use:**
   ```bash
   streamlit run ui/app.py --server.port 8502
   ```

2. **Missing Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Search Index Not Found:**
   - The UI will automatically build the index on first run
   - This may take a few minutes

4. **Slow Performance:**
   - Check available memory
   - Reduce dataset size for testing
   - Use caching for repeated operations

### Support
- Check the main project README
- Review error messages in the terminal
- Ensure all dependencies are installed
- Verify Python version compatibility

## 📈 Performance Metrics

- **Search Speed**: < 100ms for typical queries
- **Visualization Rendering**: < 2s for complex plots
- **Transformation Time**: < 5s for 5-step paths
- **Memory Usage**: < 500MB for full dataset
- **Startup Time**: < 30s for first run

## 🎉 Getting Started

1. **Quick Start:**
   ```bash
   python run_ui.py
   ```

2. **Explore Features:**
   - Try semantic search with different queries
   - Experiment with code transformations
   - Explore the embedding space visualizations
   - Run performance benchmarks

3. **Customize:**
   - Modify the UI code for your needs
   - Add new visualization types
   - Integrate with your own datasets
   - Extend the search functionality

Enjoy exploring the Holistic Semantic Continuum! 🧠✨ 