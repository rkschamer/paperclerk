"""Synthetische Testdaten: deutsche Beispieldokumente als Anhänge in einer MBOX-Datei."""

from __future__ import annotations

import io
import zipfile
from email.message import EmailMessage
from email.utils import format_datetime, make_msgid
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
from PIL import Image

INVOICE = """Telekom Deutschland GmbH · Landgrabenweg 151 · 53227 Bonn

Herrn
Max Mustermann
Musterstraße 1
12345 Musterstadt

Rechnung
Rechnungsdatum: 17.04.2023
Kundennummer: 123 456 7890
Rechnungsnummer: 4400 1234 5678
Leistungszeitraum: 01.04.2023 - 30.04.2023

Beträge in Euro
MagentaMobil M                         39,95
Telefonie Inland                        4,20
Summe netto                            37,10
Umsatzsteuer 19 %                       7,05
Rechnungsbetrag                        44,15

Der Rechnungsbetrag wird am 28.04.2023 von Ihrem Konto DE12 3456 abgebucht."""

AGB = """Allgemeine Geschäftsbedingungen der Muster Shop GmbH

§ 1 Geltungsbereich
(1) Diese Allgemeinen Geschäftsbedingungen gelten für alle Bestellungen, die Verbraucher über unseren
Onlineshop abschließen. (2) Abweichende Bedingungen des Kunden werden nicht anerkannt.

§ 2 Vertragsschluss
(1) Die Darstellung der Produkte im Onlineshop stellt kein rechtlich bindendes Angebot dar.
(2) Durch Anklicken des Buttons "Jetzt kaufen" geben Sie eine verbindliche Bestellung ab.

§ 3 Preise und Versandkosten
Die angegebenen Preise enthalten die gesetzliche Umsatzsteuer. Versandkosten werden gesondert ausgewiesen.

§ 4 Eigentumsvorbehalt
Die Ware bleibt bis zur vollständigen Bezahlung unser Eigentum.

§ 5 Gewährleistung
Es gelten die gesetzlichen Gewährleistungsrechte."""

WIDERRUF = """Widerrufsbelehrung

Widerrufsrecht
Sie haben das Recht, binnen vierzehn Tagen ohne Angabe von Gründen diesen Vertrag zu widerrufen.
Die Widerrufsfrist beträgt vierzehn Tage ab dem Tag, an dem Sie oder ein von Ihnen benannter Dritter
die Waren in Besitz genommen haben. Um Ihr Widerrufsrecht auszuüben, müssen Sie uns mittels einer
eindeutigen Erklärung über Ihren Entschluss informieren.

Folgen des Widerrufs
Wenn Sie diesen Vertrag widerrufen, haben wir Ihnen alle Zahlungen unverzüglich zurückzuzahlen.

Muster-Widerrufsformular
(Wenn Sie den Vertrag widerrufen wollen, dann füllen Sie bitte dieses Formular aus.)
An Muster Shop GmbH, Shopweg 3, 10115 Berlin
Hiermit widerrufe(n) ich/wir den von mir/uns abgeschlossenen Vertrag über den Kauf der folgenden Waren:"""

SCAN_INVOICE = """Malerbetrieb Schulze
Meisterbetrieb seit 1987 - Farbweg 7 - 12345 Musterstadt

Herrn Max Mustermann, Musterstraße 1, 12345 Musterstadt

RECHNUNG Nr. 2023-118
Datum: 05.06.2023

Pos. 1  Malerarbeiten Wohnzimmer, 42 m2        980,00 EUR
Pos. 2  Material (Farbe, Abdeckung)             270,00 EUR
Nettobetrag                                   1.250,00 EUR
MwSt. 19 %                                      237,50 EUR
Gesamtbetrag                                  1.487,50 EUR

Zahlbar innerhalb von 14 Tagen ohne Abzug."""

POLICE = """HUK-COBURG Versicherungsgruppe · 96444 Coburg

Versicherungsschein Nr. 123/456789-K
Kraftfahrtversicherung

Versicherungsnehmer: Max Mustermann, Musterstraße 1, 12345 Musterstadt
Fahrzeug: VW Golf, amtliches Kennzeichen M-AB 1234
Versicherungsbeginn: 01.01.2024, Hauptfälligkeit 01.01.
Kfz-Haftpflicht, Vollkasko mit 500 EUR Selbstbeteiligung
Jahresbeitrag: 412,30 EUR

Coburg, den 12.12.2023"""

WERBUNG = """Frühlingsangebote bei Möbel Muster!

Bis zu 50 % Rabatt auf ausgewählte Sofas, Sessel und Esstische.
Nur für kurze Zeit: Mit dem Gutscheincode FRUEHLING24 sparen Sie zusätzlich 10 %.
Jetzt im Onlineshop entdecken oder in einer unserer 40 Filialen vorbeischauen.
Kostenlose Lieferung ab 500 EUR Bestellwert. Sie erhalten diesen Newsletter, weil Sie sich angemeldet haben."""

DATENSCHUTZ = """Datenschutzhinweise gemäß Art. 13 DSGVO

1. Verantwortlicher
Verantwortlich für die Datenverarbeitung ist die Stadtwerke Musterstadt GmbH.
2. Zwecke und Rechtsgrundlagen der Verarbeitung
Wir verarbeiten Ihre personenbezogenen Daten zur Erfüllung des Vertrags (Art. 6 Abs. 1 lit. b DSGVO).
3. Speicherdauer
Ihre Daten werden gelöscht, sobald sie für die genannten Zwecke nicht mehr erforderlich sind.
4. Ihre Rechte
Sie haben das Recht auf Auskunft, Berichtigung, Löschung und Einschränkung der Verarbeitung."""

PIB = """Produktinformationsblatt zur Hausratversicherung

Welche Art von Versicherung ist das?
Wir bieten Ihnen eine Hausratversicherung an. Sie schützt Ihren Hausrat gegen Feuer, Leitungswasser,
Sturm, Hagel sowie Einbruchdiebstahl.
Was ist versichert? Der gesamte Hausrat in Ihrer Wohnung, zum Beispiel Möbel und Kleidung.
Was ist nicht versichert? Gebäudebestandteile, Kraftfahrzeuge.
Wo bin ich versichert? In der im Versicherungsschein genannten Wohnung."""

LOHN = """Muster AG - Entgeltabrechnung März 2024

Personalnummer 4711 - Max Mustermann - Steuerklasse 1
Bruttobezüge                       4.500,00
Lohnsteuer                           712,41
Solidaritätszuschlag                   0,00
Sozialversicherung                   897,47
Nettobezüge                        2.890,12
Auszahlung auf DE12 3456 7890 0000 0000 00"""

BESTELLUNG = """Bestellbestätigung

Vielen Dank für Ihre Bestellung bei Technik Direkt!
Bestellnummer: TD-2023-884512, Bestelldatum: 10.11.2023
Lieferadresse: Max Mustermann, Musterstraße 1, 12345 Musterstadt

1 x Kopfhörer Sony WH-1000XM5           299,00 EUR
Versand                                    0,00 EUR
Gesamt                                   299,00 EUR
Zahlungsart: PayPal"""

RECEIPT = """MediaMarkt Musterstadt
Am Markt 1, 12345 Musterstadt

Kaffeevollautomat EQ.6       549,00
Entkalker                     12,99
SUMME EUR                    561,99
Gegeben EC-Karte             561,99
MwSt 19% Netto 472,26 Steuer 89,73
03.02.2024 14:12  Bon 4711  Kasse 3
Vielen Dank für Ihren Einkauf!"""


def text_pdf(text: str, fontsize: float = 10) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(60, 60, 540, 800), text, fontsize=fontsize, fontname="helv")
    return doc.tobytes()


def _render(text: str, width: float = 595, height: float = 842, fontsize: float = 11, dpi: int = 150) -> Image.Image:
    doc = pymupdf.open()
    page = doc.new_page(width=width, height=height)
    page.insert_textbox(pymupdf.Rect(30, 30, width - 30, height - 30), text, fontsize=fontsize, fontname="helv")
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def scanned_pdf(text: str) -> bytes:
    im = _render(text)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=buf.getvalue())
    return doc.tobytes()


def receipt_jpg(text: str) -> bytes:
    im = _render(text, width=260, height=330, fontsize=10, dpi=220)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def logo_png() -> bytes:
    im = Image.new("RGB", (120, 40), (200, 30, 30))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def encrypted_pdf(text: str) -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), text)
    return doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="1234", owner_pw="geheim")


def zipped(name: str, data: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, data)
    return buf.getvalue()


def mail(sender: str, subject: str, date: str, body: str, attachments: list[tuple], labels: str = "Posteingang"):
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = "Max Mustermann <max@example.org>"
    msg["Subject"] = subject
    msg["Date"] = format_datetime(datetime.fromisoformat(date).replace(tzinfo=timezone.utc))
    msg["Message-ID"] = make_msgid(domain="example.org")
    msg["X-Gmail-Labels"] = labels
    msg.set_content(body)
    for att in attachments:
        data, maintype, subtype, filename = att[:4]
        extra = att[4] if len(att) > 4 else {}
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename, **extra)
    return msg


def build_messages() -> list[EmailMessage]:
    invoice = text_pdf(INVOICE)
    shop_terms = [(text_pdf(AGB), "application", "pdf", "AGB_Muster_Shop.pdf"),
                  (text_pdf(WIDERRUF), "application", "pdf", "Widerrufsbelehrung.pdf")]
    msgs = [
        mail("Telekom <rechnung@telekom.de>", "Ihre Rechnung April 2023", "2023-04-18T08:00:00",
             "Guten Tag, anbei erhalten Sie Ihre aktuelle Rechnung.",
             [(invoice, "application", "pdf", "Rechnung_2023_04.pdf")]),
        mail("Muster Shop <service@mustershop.de>", "Ihre Bestellung", "2023-11-10T12:00:00",
             "Vielen Dank für Ihre Bestellung. Im Anhang finden Sie unsere AGB und die Widerrufsbelehrung.",
             [*shop_terms, (text_pdf(BESTELLUNG), "application", "pdf", "Bestellbestaetigung.pdf")]),
        mail("Muster Shop <service@mustershop.de>", "Ihre zweite Bestellung", "2023-12-01T12:00:00",
             "Danke für Ihren Einkauf.", shop_terms),
        mail("Malerbetrieb Schulze <info@maler-schulze.de>", "Rechnung", "2023-06-06T09:30:00",
             "Hallo Herr Mustermann, anbei die Rechnung für die Arbeiten. Viele Grüße",
             [(scanned_pdf(SCAN_INVOICE), "application", "octet-stream", "scan0001.pdf")]),
        mail("HUK-COBURG <service@huk-coburg.de>", "Ihre Unterlagen zur Kfz-Versicherung", "2023-12-13T10:00:00",
             "Sehr geehrter Herr Mustermann, anbei Ihr Versicherungsschein.",
             [(text_pdf(POLICE), "application", "pdf", "Versicherungsschein.pdf"),
              (text_pdf(PIB), "application", "pdf", "Produktinformationsblatt.pdf")]),
        mail("Möbel Muster <newsletter@moebel-muster.de>", "Frühlingsangebote!", "2024-03-20T07:00:00",
             "Unsere aktuellen Angebote im Anhang.",
             [(text_pdf(WERBUNG, 14), "application", "pdf", "Prospekt_Fruehling.pdf"),
              (logo_png(), "image", "png", "logo.png", {"disposition": "inline", "cid": "<logo1@moebel>"})]),
        mail("Stadtwerke Musterstadt <kundenservice@sw-musterstadt.de>", "Ihr Stromvertrag", "2024-01-05T11:00:00",
             "Anbei unsere Datenschutzhinweise.", [(text_pdf(DATENSCHUTZ), "application", "pdf", "Datenschutz.pdf")]),
        mail("Max Mustermann <max@example.org>", "Kassenbon Kaffeemaschine", "2024-02-03T15:00:00",
             "Foto vom Bon für die Garantie.", [(receipt_jpg(RECEIPT), "image", "jpeg", "IMG_2041.jpg")]),
        mail("Lohnbuchhaltung Muster AG <lohn@muster-ag.de>", "Entgeltabrechnung 03/2024", "2024-03-28T16:00:00",
             "Ihre Abrechnung als ZIP.",
             [(zipped("Abrechnung_2024_03.pdf", text_pdf(LOHN)), "application", "zip", "Abrechnung.zip"),
              (b"\x30\x82\x01\x00" + b"\x00" * 300, "application", "pkcs7-signature", "smime.p7s")]),
        mail("Telekom <rechnung@telekom.de>", "Erinnerung: Ihre Rechnung", "2023-05-02T08:00:00",
             "Zur Erinnerung erneut Ihre Rechnung.", [(invoice, "application", "pdf", "Rechnung_2023_04.pdf")]),
        mail("Bank <noreply@bank.example>", "Ihr Kontoauszug", "2024-04-01T06:00:00",
             "Ihr Kontoauszug ist mit Ihrem Geburtsdatum geschützt.",
             [(encrypted_pdf("Kontoauszug 03/2024"), "application", "pdf", "Kontoauszug.pdf")]),
        mail("Gewinnspiel <win@spam.example>", "Sie haben gewonnen!", "2024-04-02T06:00:00",
             "Öffnen Sie den Anhang.", [(text_pdf("Gewinn!"), "application", "pdf", "gewinn.pdf")],
             labels="Spam"),
        mail("Freund <freund@example.org>", "Hallo", "2024-04-03T06:00:00", "Nur ein kurzer Gruß ohne Anhang.", []),
    ]
    return msgs


def write_mbox(path: Path, messages: list[EmailMessage] | None = None) -> Path:
    messages = messages if messages is not None else build_messages()
    with path.open("wb") as f:
        for i, msg in enumerate(messages):
            f.write(f"From 17{i:017d}@xxx Mon Jan 01 00:00:00 +0000 2024\n".encode())
            f.write(msg.as_bytes())
            f.write(b"\n")
    return path


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1] if len(sys.argv) > 1 else "test.mbox")
    write_mbox(target)
    print(f"{target} geschrieben")
