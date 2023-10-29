import { Link } from "react-router-dom" ;
import Navbar from "./Navbar";

function Home() {
    return (
        <div>
            <Navbar />
            <div className="content_container">
                <div className="content">
                    <h1>Soil Classifier</h1>
                    <h4>Classify Soil Type For Better Crops</h4>
                    <Link className="navbar_links" to="/classifier">Try Now!</Link>
                </div>
            </div>
        </div>
    )
}

export default Home;

// #678B98, #987467, #503D36, #364950