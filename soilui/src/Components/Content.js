import { Link } from "react-router-dom" 

function Content() {
    return (
        <div style={{width: "700px"}}className="content_container">
            <div className="content">
                <h1>Soil Classifier</h1>
                <h4>Classify Soil Type So Better Crops</h4>
                <Link className="navbar_links" to="/classifier">Try Now!</Link>
            </div>
        </div>
    )
}

export default Content;