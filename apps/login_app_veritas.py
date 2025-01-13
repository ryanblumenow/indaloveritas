import time
import streamlit as st

class LoginApp:
    """
    A login application to secure access within a Streamlit application.
    This implementation uses session state to manage user login and redirection.
    """

    def __init__(self, title='', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self) -> None:
        """
        Application entry point.
        """

        st.markdown("<h1 style='text-align: center;'>Please log in</h1>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([2, 2, 2])
        
        form_data = self._create_login_form(c2)

        pretty_btn = """
        <style>
        div[class="row-widget stButton"] > button {
            width: 100%;
        }
        </style>
        <br><br>
        """
        c2.markdown(pretty_btn, unsafe_allow_html=True)

        if form_data['submitted']:
            self._do_login(form_data, c2)

    def _create_login_form(self, parent_container) -> dict:
        """
        Create a login form inside the given container.
        """
        login_form = parent_container.form(key="login_form")

        form_state = {}
        form_state['username'] = login_form.text_input('Username')
        form_state['password'] = login_form.text_input('Password', type="password")
        form_state['submitted'] = login_form.form_submit_button('Login')

        if parent_container.button('Guest Login', key='guestbtn'):
            # Set guest access in session state
            st.session_state['logged_in'] = True
            st.session_state['username'] = 'guest'
            st.session_state['access_level'] = 1
            st.experimental_rerun()

        if parent_container.button('Sign Up', key='signupbtn'):
            # Redirect to a sign-up page (or handle sign-up logic here)
            st.warning("Sign-up functionality is not implemented yet.")

        return form_state

    def _do_login(self, form_data, msg_container) -> None:
        """
        Handle login logic and update session state on success.
        """
        access_level = self._check_login(form_data)

        if access_level > 0:
            msg_container.success("✔️ Login successful")
            with st.spinner("Redirecting to application..."):
                time.sleep(1)

                # Update session state for successful login
                st.session_state['logged_in'] = True
                st.session_state['username'] = form_data['username']
                st.session_state['access_level'] = access_level

                st.rerun()
        else:
            st.session_state['logged_in'] = False
            st.session_state['username'] = None

            msg_container.error("❌ Login unsuccessful. Please check your username and password and try again.")

    def _check_login(self, login_data) -> int:
        """
        Validate login credentials and return access level.
        """
        if login_data['username'] == 'Ryan' and login_data['password'] == 'ryan123':
            return 1  # Default access level
        else:
            return 0
