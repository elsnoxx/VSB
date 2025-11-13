package cz.transys.moldapp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.platform.LocalContext
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import cz.transys.moldapp.ui.scanners.HoneywellScanner
import cz.transys.moldapp.ui.screens.*

// shared scanner
val LocalScanner = staticCompositionLocalOf<HoneywellScanner?> { null }
class MainActivity : ComponentActivity() {
    private lateinit var scanner: HoneywellScanner

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        scanner = HoneywellScanner(this)
        scanner.open()

        setContent {
            CompositionLocalProvider(LocalScanner provides scanner) {
                MoldAppApp()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        scanner.close()
    }
}

@Composable
fun MoldAppApp() {
    val navController = rememberNavController()
    val context = LocalContext.current

    Surface {
        NavHost(
            navController = navController,
            startDestination = "login" // ← nová startovací stránka
        ) {
            composable("login") {
                LoginScreen(navController)
            }
            composable("menu") {
                MenuScreen(context, navController)
            }
            composable("tag_write") {
                TagWriteScreen { navController.popBackStack() }
            }
            composable("mold_repair") {
                MoldRepairScreen { navController.popBackStack() }
            }
            composable("part_change") {
                PartChangeScreen { navController.popBackStack() }
            }
            composable("mold_mount") {
                MoldMountScreen { navController.popBackStack() }
            }
            composable("rf_tag_info") {
                RfTagInfoScreen { navController.popBackStack() }
            }
            composable("test_reading") {
                TestReadingScreen { navController.popBackStack() }
            }
        }
    }
}
