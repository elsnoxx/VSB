package cz.transys.moldapp.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavHostController
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.platform.LocalContext
import cz.transys.moldapp.LocalScanner
import kotlinx.coroutines.delay
import cz.transys.moldapp.ui.localdata.LocalStorage

@Composable
fun LoginScreen(navController: NavHostController) {
    var userId by remember { mutableStateOf("") }
    var errorMessage by remember { mutableStateOf("") }
    val context = LocalContext.current
    val storage = remember { LocalStorage(context) }
    val scope = rememberCoroutineScope()
    val focusRequester = remember { FocusRequester() }

    fun validateAndProceed(data: String) {
        val cleanData = data.trim()
        if (cleanData.matches(Regex("^\\d+$"))) {
            userId = cleanData
            errorMessage = ""
        } else {
            errorMessage = "Naskenovaný kód není platné User ID (povolena jsou jen čísla)"
            userId = ""
        }
    }

    val scanner = LocalScanner.current
    // 🧩 zaregistrujeme listener na sken
    LaunchedEffect(scanner) {
        scanner?.setOnScanListener { scannedData ->
            validateAndProceed(scannedData.trim())
        }
    }

    DisposableEffect(scanner) {
        onDispose {
            // po opuštění obrazovky zrušíme listener
            scanner?.setOnScanListener { }
        }
    }

    // Po načtení obrazovky nastaví fokus
    LaunchedEffect(Unit) {
        delay(300) // malá prodleva, aby se view inicializovalo
//        focusRequester.requestFocus()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFECECEC))
            .padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "🔑 MoldApp Login",
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold,
            color = Color(0xFF1565C0),
            modifier = Modifier.padding(bottom = 32.dp)
        )

        OutlinedTextField(
            value = userId,
            onValueChange = {
                userId = it
                errorMessage = ""
            },

            label = { Text("User ID") },
            placeholder = { Text("Zadej své ID") },
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            singleLine = true,
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp)
                .focusRequester(focusRequester)
        )

        if (errorMessage.isNotEmpty()) {
            Text(
                text = errorMessage,
                color = Color.Red,
                fontSize = 14.sp,
                modifier = Modifier.padding(4.dp)
            )
        }

        Spacer(modifier = Modifier.height(24.dp))

        Button(
            onClick = {
                if (userId.isBlank()) {
                    errorMessage = "Prosím, zadej své User ID"
                } else {
                    // TODO: zde můžeš později ověřit ID z API
                    storage.saveUserId(userId)
                    navController.navigate("menu")
                }
            },
            modifier = Modifier
                .fillMaxWidth()
                .height(60.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = Color(0xFF1565C0),
                contentColor = Color.White
            )
        ) {
            Text(
                text = "Pokračovat",
                fontWeight = FontWeight.Bold,
                fontSize = 20.sp
            )
        }
    }
}