package cz.transys.moldapp.ui.screens

import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cz.transys.moldapp.LocalScanner
import cz.transys.moldapp.R
import cz.transys.moldapp.buisines.apicalls.moldapi.CarrierMountResponse
import cz.transys.moldapp.buisines.apicalls.moldapi.CarriersList
import cz.transys.moldapp.buisines.apicalls.moldapi.MoldApiRepository
import cz.transys.moldapp.buisines.apicalls.moldmount.MoldMountRepository
import cz.transys.moldapp.buisines.apicalls.moldmount.MoldMountResquest
import cz.transys.moldapp.buisines.apicalls.rfidinfo.MoldRfInfoRepository
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MoldMountScreen(onBack: () -> Unit) {
    val colors = MaterialTheme.colorScheme
    val scope = rememberCoroutineScope()
    val repo = remember { MoldApiRepository() }
    val rfRepo = remember { MoldRfInfoRepository() }
    val mountRepo = remember { MoldMountRepository() }

    var showSuccessDialog by remember { mutableStateOf(false) }
    var showErrorDialog by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf("") }


    var showDuplicateDialog by remember { mutableStateOf(false) }

    val scanner = LocalScanner.current

    // Carrier dropdown
    var expandedCarrier by remember { mutableStateOf(false) }
    var selectedCarrier by remember { mutableStateOf("") }

    var carrierList by remember { mutableStateOf<List<CarriersList>>(emptyList()) }
    var mountInfo by remember { mutableStateOf<CarrierMountResponse?>(null) }

    // MOLD #1
    var mold1Type by remember { mutableStateOf("") }
    var mold1Code by remember { mutableStateOf("") }
    var carCode1 by remember { mutableStateOf("") }

    // MOLD #2
    var mold2Type by remember { mutableStateOf("") }
    var mold2Code by remember { mutableStateOf("") }
    var carCode2 by remember { mutableStateOf("") }

    fun sendMountRequest() {
        if (selectedCarrier.isBlank()) {
            errorMessage = "Select carrier first"
            showErrorDialog = true
            return
        }
        if (mold1Code.isBlank() || mold2Code.isBlank()) {
            errorMessage = "Scan both molds first"
            showErrorDialog = true
            return
        }

        val request = MoldMountResquest(
            carrirer_no = selectedCarrier,
            carrirer_name = carrierList.find { it.code_value1 == selectedCarrier }?.code_value2
                ?: "",
            car_code1 = carCode1,
            car_code2 = carCode2,
            mold_code1 = mold1Code,
            mold_code2 = mold2Code,
            type1 = mold1Type,
            type2 = mold2Type
        )

        scope.launch {
            try {
                val success = mountRepo.postMoldMount(request)

                if (success) {
                    showSuccessDialog = true

                    // Reset po úspěchu
                    mold1Code = ""
                    mold1Type = ""
                    carCode1 = ""

                    mold2Code = ""
                    mold2Type = ""
                    carCode2 = ""

                } else {
                    errorMessage = "Server returned error"
                    showErrorDialog = true
                }
            } catch (e: Exception) {
                errorMessage = e.localizedMessage ?: "Unknown error"
                showErrorDialog = true
            }
        }
    }

    if (showSuccessDialog) {
        AlertDialog(
            onDismissRequest = { showSuccessDialog = false },
            title = { Text(stringResource(R.string.success_sent)) },
            text = { Text(stringResource(R.string.dialog_mold_mount_text_succes)) },
            confirmButton = {
                TextButton(onClick = { showSuccessDialog = false }) {
                    Text(stringResource(R.string.dialog_mold_mount_button))
                }
            }
        )
    }

    if (showErrorDialog) {
        AlertDialog(
            onDismissRequest = { showErrorDialog = false },
            title = { Text("Error") },
            text = { Text(errorMessage) },
            confirmButton = {
                TextButton(onClick = { showErrorDialog = false }) {
                    Text(stringResource(R.string.dialog_mold_mount_button))
                }
            }
        )
    }



    LaunchedEffect(scanner, selectedCarrier) {
        scanner?.setOnScanListener { scannedTag ->

            val tag = scannedTag.trim()

            // NESMÍ být prázdný
            if (tag.isBlank()) return@setOnScanListener

            if (tag == mold1Code || tag == mold2Code) {

                showDuplicateDialog = true

                scope.launch {
                    showDuplicateDialog = false
                }

                return@setOnScanListener
            }


            // LOAD TAG INFO FROM API
            scope.launch {
                val info = rfRepo.getTagInfo(tag)

                if (info != null) {

                    // FIRST SLOT EMPTY? → FILL MOLD #1
                    if (mold1Code.isBlank()) {
                        mold1Code = info.mold_code
                        mold1Type = info.mold_name
                        carCode1 = info.car_code
                    }

                    // SECOND SLOT EMPTY? → FILL MOLD #2
                    else if (mold2Code.isBlank()) {
                        mold2Code = info.mold_code
                        mold2Type = info.mold_name
                        carCode2 = info.car_code
                    }

                } else {
                    Log.e("RFID", "Tag not found in API")
                }
            }
        }
    }

    DisposableEffect(scanner) {
        onDispose { scanner?.setOnScanListener { } }
    }


    // Load carriers
    LaunchedEffect(Unit) {
        try {
            carrierList = repo.getAllCarriers()
        } catch (_: Exception) {}
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(colors.background)
            .padding(16.dp),
        verticalArrangement = Arrangement.Top,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {

        if (showDuplicateDialog) {
            AlertDialog(
                onDismissRequest = { showDuplicateDialog = false },
                title = { Text(stringResource(R.string.dialog_mold_mount_title)) },
                text = { Text(stringResource(R.string.dialog_mold_mount_text)) },
                confirmButton = {
                    TextButton(onClick = { showDuplicateDialog = false }) {
                        Text(stringResource(R.string.dialog_mold_mount_button))
                    }
                }
            )
        }

        Text(
            text = stringResource(R.string.mold_mount_title),
            fontSize = 26.sp,
            fontWeight = FontWeight.Bold,
            color = colors.primary,
            modifier = Modifier.padding(bottom = 20.dp)
        )

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(colors.surface)
                .padding(16.dp)
        ) {

            // CARRIER SELECT
            ExposedDropdownMenuBox(
                expanded = expandedCarrier,
                onExpandedChange = { expandedCarrier = !expandedCarrier }
            ) {
                OutlinedTextField(
                    value = selectedCarrier,
                    onValueChange = {},
                    readOnly = true,
                    label = { Text(stringResource(R.string.carrier_label)) },
                    trailingIcon = {
                        ExposedDropdownMenuDefaults.TrailingIcon(expanded = expandedCarrier)
                    },
                    modifier = Modifier
                        .menuAnchor()
                        .fillMaxWidth()
                )

                ExposedDropdownMenu(
                    expanded = expandedCarrier,
                    onDismissRequest = { expandedCarrier = false }
                ) {
                    if (carrierList.isEmpty()) {
                        DropdownMenuItem(
                            text = { Text("No carriers loaded") },
                            onClick = {}
                        )
                    } else {
                        carrierList.forEach { carrier ->
                            DropdownMenuItem(
                                text = { Text(carrier.code_value2) },
                                onClick = {
                                    selectedCarrier = carrier.code_value1
                                    expandedCarrier = false
                                }
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // MOLD #1 – read only
            MoldMountSection(
                title = stringResource(R.string.mold1_title),
                color = colors.tertiary,
                carCode = carCode1,
                type = mold1Type,
                code = mold1Code,
                readOnlyCode = true,
                onCodeChange = {},
                onClear = {
                    mold1Code = ""
                    mold1Type = ""
                    carCode1 = ""
                }
            )

            Spacer(modifier = Modifier.height(12.dp))

            // MOLD #2 – editable (new mold to mount)
            MoldMountSection(
                title = stringResource(R.string.mold2_title),
                color = colors.error,
                carCode = carCode2,
                type = mold2Type,
                code = mold2Code,
                readOnlyCode = false,
                onCodeChange = { mold2Code = it },onClear = {
                    mold2Code = ""
                    mold2Type = ""
                    carCode2 = ""
                }
            )

            Spacer(modifier = Modifier.height(20.dp))

            Button(
                onClick = { sendMountRequest() },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(60.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = colors.primary,
                    contentColor = colors.onPrimary
                )
            ) {
                Text(
                    text = stringResource(R.string.mount_button),
                    fontWeight = FontWeight.Bold,
                    fontSize = 20.sp
                )
            }


            Spacer(modifier = Modifier.height(16.dp))

            Button(
                onClick = onBack,
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = colors.secondary,
                    contentColor = colors.onSecondary
                )
            ) {
                Text(stringResource(R.string.back_button))
            }
        }
    }
}

@Composable
fun MoldMountSection(
    title: String,
    color: Color,
    carCode: String,
    type: String,
    code: String,
    readOnlyCode: Boolean,
    onCodeChange: (String) -> Unit,
    onClear: () -> Unit // <-- přidané
) {

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(color.copy(alpha = 0.1f))
            .padding(12.dp)
    ) {

        Text(
            text = title,
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold,
            color = color
        )

        Spacer(modifier = Modifier.height(6.dp))

        OutlinedTextField(
            value = code,
            onValueChange = { if (!readOnlyCode) onCodeChange(it) },
            readOnly = readOnlyCode,
            label = { Text("Mold code") },
            trailingIcon = {
                if (code.isNotBlank()) {
                    IconButton(onClick = { onClear() }) {
                        Icon(
                            imageVector = Icons.Default.Delete,
                            contentDescription = "Clear",
                            tint = Color.Red
                        )
                    }
                }
            },
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(6.dp))

        Text("Type: $type")
        Text("Car: $carCode")
    }
}