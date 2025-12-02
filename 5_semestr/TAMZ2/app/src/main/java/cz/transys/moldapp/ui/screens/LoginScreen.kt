package cz.transys.moldapp.ui.screens

import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavHostController
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import cz.transys.moldapp.LocalScanner
import cz.transys.moldapp.R
import cz.transys.moldapp.buisines.apicalls.moldapi.EmpIdResponse
import cz.transys.moldapp.buisines.apicalls.moldapi.MoldApiRepository
import cz.transys.moldapp.buisines.localdata.LocalStorage
import kotlinx.coroutines.launch

@Composable
fun LoginScreen(navController: NavHostController) {
    val colors = MaterialTheme.colorScheme
    val repo = remember { MoldApiRepository() }
    val logInfo = remember { mutableStateOf<EmpIdResponse?>(null) }

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
            errorMessage = context.getString(R.string.invalid_user_id)
            userId = ""
        }
    }

    val scanner = LocalScanner.current

    LaunchedEffect(scanner) {
        scanner?.setOnScanListener { scannedData ->
            validateAndProceed(scannedData.trim())
        }
    }

    DisposableEffect(scanner) {
        onDispose { scanner?.setOnScanListener { } }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(colors.background)
            .padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = stringResource(R.string.moldapp_login_title),
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold,
            color = colors.primary,
            modifier = Modifier.padding(bottom = 32.dp)
        )

        OutlinedTextField(
            value = userId,
            onValueChange = {
                userId = it
                errorMessage = ""
            },
            label = { Text(stringResource(R.string.user_id)) },
            placeholder = { Text(stringResource(R.string.user_id_placeholder)) },
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            singleLine = true,
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp)
                .focusRequester(focusRequester),
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = colors.primary,
                unfocusedBorderColor = colors.outline,
                focusedLabelColor = colors.primary,
                cursorColor = colors.primary
            )
        )

        if (errorMessage.isNotEmpty()) {
            Text(
                text = errorMessage,
                color = colors.error,
                fontSize = 14.sp,
                modifier = Modifier.padding(4.dp)
            )
        }

        Spacer(modifier = Modifier.height(24.dp))

        Button(
            onClick = {
                if (userId.isBlank()) {
                    errorMessage = context.getString(R.string.empty_user_id)
                    return@Button
                }

                scope.launch {
                    try {
                        val response = repo.login(userId)
                        logInfo.value = response
                        Log.d("Login API", "Login success: $response")

                        if (response.message != "issue" && response.emp_name != ""){
                            storage.saveUserId(userId)
                            navController.navigate("menu")
                        }  else {
                            errorMessage = context.getString(R.string.wrong_user_id)
                        }
                    } catch (e: Exception) {
                        Log.e("Login API", "Error: ${e.localizedMessage}")
                        errorMessage = context.getString(R.string.api_error_full)
                    }
                }
            },
            modifier = Modifier
                .fillMaxWidth()
                .height(60.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = colors.primary,          // ← theme primary
                contentColor = colors.onPrimary           // ← text kontrast
            )
        ) {
            Text(
                text = stringResource(R.string.continue_button),
                fontWeight = FontWeight.Bold,
                fontSize = 20.sp
            )
        }
    }
}
