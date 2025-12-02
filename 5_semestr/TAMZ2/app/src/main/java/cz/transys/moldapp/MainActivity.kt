package cz.transys.moldapp

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.lazy.layout.IntervalList
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.platform.LocalContext
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import cz.transys.moldapp.buisines.apicalls.moldapi.CarCodeList
import cz.transys.moldapp.buisines.apicalls.moldapi.CarriersList
import cz.transys.moldapp.buisines.apicalls.moldapi.MoldApiRepository
import cz.transys.moldapp.buisines.apicalls.moldrepair.MoldRepairRepository
import cz.transys.moldapp.buisines.apicalls.moldrepair.RepairTypes
import cz.transys.moldapp.buisines.models.ConnectivityStatus
import cz.transys.moldapp.buisines.scanners.HoneywellScanner
import cz.transys.moldapp.ui.screens.*
import cz.transys.moldapp.ui.theme.MoldAppTheme


// shared scanner
val LocalScanner = staticCompositionLocalOf<HoneywellScanner?> { null }
val LocalConnectivity = staticCompositionLocalOf {
    ConnectivityStatus(internet = true, apiAvailable = true)
}

class MainActivity : ComponentActivity() {

    private lateinit var scanner: HoneywellScanner
    private lateinit var connectivityManager: ConnectivityManager



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        scanner = HoneywellScanner(this)
        scanner.open()

        connectivityManager =
            getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager



        setContent {
            val isConnected = connectivityState(connectivityManager)

            isConnected.internet

            CompositionLocalProvider(
                LocalScanner provides scanner,
                LocalConnectivity provides isConnected
            ) {
                MoldAppTheme {
                    AppRoot()
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        scanner.close()
    }
}


@Composable
fun AppRoot() {
    val navController = rememberNavController()
    val status = LocalConnectivity.current
    val context = LocalContext.current
    var showInternetDialog by remember { mutableStateOf(false) }
    var showApiDialog by remember { mutableStateOf(false) }

    val repo = remember { MoldApiRepository() }
    val repairRepo = remember { MoldRepairRepository() }

    var carrierList by remember { mutableStateOf<List<CarriersList>>(emptyList()) }
    var typeList by remember { mutableStateOf<List<RepairTypes>>(emptyList()) }
    var carList by remember { mutableStateOf<List<CarCodeList>>(emptyList()) }


    LaunchedEffect(status.internet) {
        if (!status.internet) showInternetDialog = true
    }

    LaunchedEffect(status.apiAvailable) {
        if (!status.internet) {
            showInternetDialog = true
            return@LaunchedEffect
        }

        if (!status.apiAvailable) {
            showApiDialog = true
            return@LaunchedEffect
        }

        try {
            carrierList = repo.getAllCarriers(forceRefresh = true)
            typeList = repairRepo.getAllRepairTypes(forceRefresh = true)
            carList = repo.getAllCars(forceRefresh = true)
        } catch (e: Exception) {
            showApiDialog = true
        }
    }


//    if (!status.internet) showInternetDialog = true
//    if (!status.apiAvailable) showApiDialog = true


    if (showInternetDialog) {
        AlertDialog(
            onDismissRequest = { showInternetDialog = false },
            title = { Text(context.getString(R.string.no_connection)) },
            text = { Text(context.getString(R.string.no_connection_text)) },
            confirmButton = {
                TextButton(
                    onClick = { showInternetDialog = false }
                ) {
                    Text("OK")
                }
            }
        )
    }


    if (showApiDialog) {
        AlertDialog(
            onDismissRequest = { showApiDialog = false },
            title = { Text(context.getString(R.string.no_connection_to_server)) },
            text = { Text(context.getString(R.string.no_connection_text_to_server)) },
            confirmButton = {
                TextButton(
                    onClick = { showApiDialog = false }
                ) {
                    Text("OK")
                }
            }
        )
    }



    Surface {
        NavHost(
            navController = navController,
            startDestination = "login"
        ) {
            composable("login") { LoginScreen(navController) }
            composable("menu") { MenuScreen(LocalContext.current, navController) }
            composable("tag_write") { TagWriteScreen { navController.popBackStack() } }
            composable("mold_repair") { MoldRepairScreen { navController.popBackStack() } }
            composable("part_change") { PartChangeScreen { navController.popBackStack() } }
            composable("mold_mount") { MoldMountScreen { navController.popBackStack() } }
            composable("rf_tag_info") { RfTagInfoScreen { navController.popBackStack() } }
            composable("test_reading") { TestReadingScreen { navController.popBackStack() } }
        }
    }
}


@Composable
fun connectivityState(
    connectivityManager: ConnectivityManager
): ConnectivityStatus {

    fun hasInternet(connectivityManager: ConnectivityManager): Boolean {
        val network = connectivityManager.activeNetwork ?: return false
        val caps = connectivityManager.getNetworkCapabilities(network) ?: return false
        return caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
    }


    val repo = remember { MoldApiRepository() }
    var internet by remember { mutableStateOf(hasInternet(connectivityManager)) }
    var apiAvailable by remember { mutableStateOf(true) }

    // checking internet connection
    LaunchedEffect(Unit) {
        val request = NetworkRequest.Builder().build()
        connectivityManager.registerNetworkCallback(
            request,
            object : ConnectivityManager.NetworkCallback() {
                override fun onAvailable(network: Network) {
                    internet = true
                }

                override fun onLost(network: Network) {
                    internet = false
                }
            }
        )
    }

    // test API every 5 sec
    LaunchedEffect(Unit) {
        while (true) {
            if (internet) {
                try {
                    apiAvailable = repo.checkApiAvailable()
                } catch (e: Exception) {
                    apiAvailable = false
                }
            } else {
                apiAvailable = false
            }
            kotlinx.coroutines.delay(5000)
        }
    }

    return ConnectivityStatus(internet = internet, apiAvailable = apiAvailable)
}
