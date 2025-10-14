package cz.transys.moldapp.ui.screens

import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import cz.transys.moldapp.MoldAppApp
import cz.transys.moldapp.ui.theme.MoldAppTheme
import androidx.activity.ComponentActivity

class LoginScreen : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MoldAppTheme {
                MoldAppApp()
            }
        }
    }
}