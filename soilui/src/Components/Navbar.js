import { Link, Outlet } from "react-router-dom"

function Navbar() {
    return (
        <>
            <div className="navbar_container">
                <div className="navbar">
                    <Link className="navbar_links" to="/">Home</Link>
                    <Link className="navbar_links" to="/classifier">Machine Learning Classifier</Link>
                </div>
            </div>
        </>
    )
}

export default Navbar;