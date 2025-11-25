package cz.transys.moldapp.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cz.transys.moldapp.R
import cz.transys.moldapp.ui.apicalls.CarriersList
import cz.transys.moldapp.ui.apicalls.MoldApiRepository


@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PartChangeScreen(onBack: () -> Unit) {
    val colors = MaterialTheme.colorScheme

    val repo = remember { MoldApiRepository() }

    // Carrier dropdown
    var expandedCarrier by remember { mutableStateOf(false) }
    var selectedCarrier by remember { mutableStateOf("") }

    // SPRÁVNĚ — teď máš stejný model jako u MoldMount
    var carrierList by remember { mutableStateOf<List<CarriersList>>(emptyList()) }

    // Mold 1 (API loaded)
    var mold1Type by remember { mutableStateOf("") }
    var mold1Code by remember { mutableStateOf("") }

    // Mold 2 (scanned)
    var mold2Type by remember { mutableStateOf("") }
    var mold2Code by remember { mutableStateOf("") }

    // Load carriers
    LaunchedEffect(Unit) {
        try {
            carrierList = repo.getAllCarriers()
        } catch (_: Exception) { }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(colors.background)
            .padding(16.dp),
        verticalArrangement = Arrangement.Top,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Title
        Text(
            text = stringResource(R.string.part_change_title),
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

            // Carrier ComboBox
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
                        .fillMaxWidth()
                )

                ExposedDropdownMenu(
                    expanded = expandedCarrier,
                    onDismissRequest = { expandedCarrier = false }
                ) {
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

            Spacer(modifier = Modifier.height(16.dp))

            // MOLD #1 – static / loaded
            MoldPartChangeSection(
                title = stringResource(R.string.mold1_title),
                color = colors.tertiary,
                carCode = selectedCarrier,
                type = mold1Type,
                code = mold1Code,
                readOnlyCode = true,
                onCodeChange = { }
            )

            Spacer(modifier = Modifier.height(12.dp))

            // MOLD #2 – scanned
            MoldPartChangeSection(
                title = stringResource(R.string.mold2_title),
                color = colors.error,
                carCode = selectedCarrier,
                type = mold2Type,
                code = mold2Code,
                readOnlyCode = false,
                onCodeChange = { mold2Code = it }
            )

            Spacer(modifier = Modifier.height(20.dp))

            // SUBMIT
            Button(
                onClick = { /* TODO API*/ },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(60.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = colors.primary,
                    contentColor = colors.onPrimary
                )
            ) {
                Text(
                    text = stringResource(R.string.submit_button),
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
fun ReadOnlyPartFieldMini(label: String, value: String, modifier: Modifier = Modifier) {
    val colors = MaterialTheme.colorScheme

    Column(modifier = modifier) {
        Text(label, fontSize = 12.sp, color = colors.onSurfaceVariant)

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(colors.surfaceVariant, MaterialTheme.shapes.small)
                .padding(8.dp)
        ) {
            Text(
                value.ifBlank { "-" },
                color = colors.onSurface
            )
        }
    }
}


@Composable
fun MoldPartChangeSection(
    title: String,
    color: Color,
    carCode: String,
    type: String,
    code: String,
    readOnlyCode: Boolean = false,
    onCodeChange: (String) -> Unit
) {
    val colors = MaterialTheme.colorScheme

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(color.copy(alpha = 0.1f), MaterialTheme.shapes.small)
            .padding(6.dp)
    ) {
        Text(
            text = title,
            color = color,
            fontWeight = FontWeight.Bold,
            fontSize = 16.sp
        )

        Spacer(modifier = Modifier.height(6.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            ReadOnlyPartFieldMini(stringResource(R.string.car_label), carCode, Modifier.weight(0.3f))
            ReadOnlyPartFieldMini(stringResource(R.string.type_label), type, Modifier.weight(1f))
        }

        Spacer(modifier = Modifier.height(6.dp))

        OutlinedTextField(
            value = code,
            onValueChange = { if (!readOnlyCode) onCodeChange(it) },
            readOnly = readOnlyCode,
            label = { Text(stringResource(R.string.mold_code)) },
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = colors.primary,
                unfocusedBorderColor = colors.outline
            ),
            modifier = Modifier.fillMaxWidth()
        )
    }
}
