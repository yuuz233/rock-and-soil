import Navbar from "./Navbar"

function SoilClassifierPage() {
    return (
        <>
            <Navbar />
            <div className="content_container">
                <div className="content">
                    <h1>Machine Learning Classifier</h1>
                    <form>
                        <label>Picture of Soil Type:
                            <input type="file" name="soilType" className="soilTypeInput" />
                        </label>
                        <div>
                            <input type="submit" value="Submit" />
                        </div>
                    </form>
                </div>
            </div>
        </>
    )
}

export default SoilClassifierPage;