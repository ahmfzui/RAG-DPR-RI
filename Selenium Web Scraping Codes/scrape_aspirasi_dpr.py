"""
DPR RI Aspirasi Enhanced Scraper
Scrape informasi aspirasi dengan nested navigation
Level 1: Main aspirasi cards
Level 2: Sub-cards untuk partisipasi masyarakat
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
import json
import csv
import time
from datetime import datetime

class DPRAspirasiEnhancedScraper:
    def __init__(self, headless=False):
        self.setup_driver(headless)
        self.aspirasi_data = {
            'main_services': [],
            'partisipasi_detail': []
        }
        self.base_url = "https://www.dpr.go.id"
        
    def setup_driver(self, headless=False):
        """Setup Chrome WebDriver"""
        print("🤖 Setting up Chrome WebDriver...")
        
        chrome_options = Options()
        
        if headless:
            chrome_options.add_argument("--headless")
            
        # Optimization options
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # User agent
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.wait = WebDriverWait(self.driver, 30)
            print("✅ WebDriver ready")
        except Exception as e:
            print(f"❌ Failed to setup WebDriver: {e}")
            print("💡 Make sure ChromeDriver is installed")
            raise
    
    def navigate_to_homepage(self):
        """Navigate to DPR homepage"""
        print(f"🌐 Navigating to: {self.base_url}")
        
        try:
            self.driver.get(self.base_url)
            
            # Wait for page to load
            print("⏳ Waiting for homepage to load...")
            
            # Wait for navigation bar
            navbar = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "nav, header"))
            )
            
            print("✅ Homepage loaded")
            time.sleep(3)  # Additional wait for all elements
            
            return True
            
        except TimeoutException:
            print("❌ Timeout loading homepage")
            return False
        except Exception as e:
            print(f"❌ Error navigating to homepage: {e}")
            return False
    
    def find_and_click_aspirasi(self):
        """Find and click Aspirasi in navbar"""
        print("🔍 Looking for Aspirasi in navbar...")
        
        try:
            # Try multiple approaches to find Aspirasi
            aspirasi_element = None
            
            # Method 1: XPath with text content
            try:
                aspirasi_elements = self.driver.find_elements(By.XPATH, "//li[.//div[contains(text(), 'Aspirasi')]]")
                if aspirasi_elements:
                    aspirasi_element = aspirasi_elements[0]
                    print("✅ Found Aspirasi via XPath (method 1)")
            except:
                pass
            
            # Method 2: Look for any clickable element with "Aspirasi"
            if not aspirasi_element:
                try:
                    clickable_elements = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Aspirasi') and (self::a or self::button or self::li or self::div[@class*='cursor-pointer'])]")
                    if clickable_elements:
                        aspirasi_element = clickable_elements[0]
                        print("✅ Found Aspirasi via clickable element (method 2)")
                except:
                    pass
            
            if not aspirasi_element:
                print("❌ Aspirasi navbar item not found")
                return False
            
            # Click the element
            print("🖱️  Clicking Aspirasi...")
            
            # Try different click methods
            click_success = False
            for i in range(3):
                try:
                    if i == 0:
                        # Method 1: Regular click
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", aspirasi_element)
                        time.sleep(1)
                        aspirasi_element.click()
                    elif i == 1:
                        # Method 2: JavaScript click
                        self.driver.execute_script("arguments[0].click();", aspirasi_element)
                    else:
                        # Method 3: ActionChains
                        ActionChains(self.driver).move_to_element(aspirasi_element).click().perform()
                    
                    time.sleep(2)
                    
                    # Check if modal opened
                    if self.wait_for_modal("aspirasi"):
                        click_success = True
                        break
                        
                except Exception as e:
                    print(f"   ❌ Click method {i+1} failed: {e}")
                    continue
            
            return click_success
            
        except Exception as e:
            print(f"❌ Error finding/clicking Aspirasi: {e}")
            return False
    
    def wait_for_modal(self, expected_content):
        """Wait for modal to appear with expected content"""
        print(f"⏳ Waiting for modal with '{expected_content}' content...")
        
        try:
            # Look for modal dialog
            modal_selectors = [
                "[id*='headlessui-dialog-panel']",
                "[role='dialog']",
                ".modal",
                "[data-headlessui-state='open']"
            ]
            
            for selector in modal_selectors:
                try:
                    modal = self.wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    
                    # Check if this modal contains expected content
                    modal_text = modal.text.lower()
                    if expected_content.lower() in modal_text:
                        print(f"✅ Modal with '{expected_content}' detected")
                        return True
                        
                except TimeoutException:
                    continue
            
            print(f"❌ Modal with '{expected_content}' not found")
            return False
            
        except Exception as e:
            print(f"❌ Error waiting for modal: {e}")
            return False
    
    def extract_main_aspirasi_cards(self):
        """Extract main aspirasi cards from level 1 modal"""
        print("📋 Extracting main aspirasi cards...")
        
        try:
            # Find all cards in the current modal
            cards = self.find_cards_in_modal()
            
            if not cards:
                print("❌ No main aspirasi cards found")
                return []
            
            # Extract data from each card
            main_cards_data = []
            
            for i, card in enumerate(cards, 1):
                try:
                    card_data = self.extract_card_data(card, i, level="main")
                    if card_data and card_data.get('title'):
                        main_cards_data.append(card_data)
                        print(f"   ✅ {i}. {card_data['title']}")
                    else:
                        print(f"   ❌ {i}. Failed to extract card data")
                        
                except Exception as e:
                    print(f"   ❌ {i}. Error extracting card: {e}")
                    continue
            
            print(f"✅ Successfully extracted {len(main_cards_data)} main aspirasi cards")
            return main_cards_data
            
        except Exception as e:
            print(f"❌ Error extracting main aspirasi cards: {e}")
            return []
    
    def find_cards_in_modal(self):
        """Find cards in current modal"""
        card_selectors = [
            ".card",
            "[class*='card']",
            ".high-contrast-card",
            ".grid > div"
        ]
        
        cards = []
        
        for selector in card_selectors:
            try:
                found_cards = self.driver.find_elements(By.CSS_SELECTOR, selector)
                
                # Filter cards that seem to be relevant
                for card in found_cards:
                    card_text = card.text.lower()
                    if any(keyword in card_text for keyword in ['pengaduan', 'partisipasi', 'kunjungan', 'aspirasi', 'pusat', 'undang-undang']):
                        cards.append(card)
                
                if cards:
                    print(f"✅ Found {len(cards)} cards with selector: {selector}")
                    break
                    
            except NoSuchElementException:
                continue
        
        return cards
    
    def extract_card_data(self, card_element, card_number, level="main"):
        """Extract data from individual card"""
        
        card_data = {
            'title': '',
            'description': '',
            'image_url': '',
            'action_url': '',
            'action_text': '',
            'type': '',
            'level': level
        }
        
        try:
            # Extract title
            title_selectors = [
                "h2.card-title",
                ".card-title h2",
                ".card-title",
                "h2",
                "[class*='title']"
            ]
            
            for selector in title_selectors:
                try:
                    title_element = card_element.find_element(By.CSS_SELECTOR, selector)
                    title_text = title_element.text.strip()
                    if title_text and len(title_text) > 3:  # Valid title
                        card_data['title'] = title_text
                        break
                except NoSuchElementException:
                    continue
            
            # Extract description
            description_selectors = [
                ".card-body p",
                "p.font-jakarta",
                "p",
                "[class*='text'][class*='sm']"
            ]
            
            for selector in description_selectors:
                try:
                    desc_elements = card_element.find_elements(By.CSS_SELECTOR, selector)
                    for desc_element in desc_elements:
                        desc_text = desc_element.text.strip()
                        # Skip if it's the title or action text
                        if (desc_text != card_data['title'] and 
                            'kirim' not in desc_text.lower() and 
                            'buat' not in desc_text.lower() and
                            len(desc_text) > 10):
                            card_data['description'] = desc_text
                            break
                    if card_data['description']:
                        break
                except NoSuchElementException:
                    continue
            
            # Extract image URL
            try:
                img_element = card_element.find_element(By.TAG_NAME, "img")
                img_src = img_element.get_attribute('src')
                if img_src:
                    # Handle relative URLs
                    if img_src.startswith('/'):
                        img_src = self.base_url + img_src
                    card_data['image_url'] = img_src
                
                # Extract alt text for additional context
                alt_text = img_element.get_attribute('alt')
                if alt_text and not card_data['title']:
                    card_data['title'] = alt_text
                    
            except NoSuchElementException:
                pass
            
            # Extract action URL and text
            try:
                action_element = card_element.find_element(By.CSS_SELECTOR, "a.btn, .card-actions a")
                action_url = action_element.get_attribute('href')
                action_text = action_element.text.strip()
                
                # Clean action text (remove icons/extra content)
                if action_text:
                    # Remove common action prefixes/suffixes
                    action_text = action_text.replace('Kirim ', '').replace('Buat ', '').replace('Sampaikan ', '')
                    card_data['action_text'] = action_text
                
                if action_url and action_url != 'javascript:void(0)' and action_url != '#':
                    card_data['action_url'] = action_url
                else:
                    card_data['action_url'] = None  # Will be handled as nested navigation
                    
            except NoSuchElementException:
                pass
            
            # Determine type based on title/content
            title_lower = card_data['title'].lower()
            desc_lower = card_data['description'].lower()
            
            if 'pengaduan' in title_lower and 'dpr' in title_lower:
                card_data['type'] = 'pengaduan_dpr'
            elif 'pengaduan' in title_lower and ('pemerintah' in title_lower or 'lapor' in desc_lower):
                card_data['type'] = 'pengaduan_pemerintah'
            elif 'partisipasi' in title_lower:
                card_data['type'] = 'partisipasi_masyarakat'
            elif 'kunjungan' in title_lower:
                card_data['type'] = 'kunjungan_masyarakat'
            elif 'politik' in title_lower and 'hukum' in title_lower:
                card_data['type'] = 'puu_polhukham'
            elif 'ekonomi' in title_lower and 'keuangan' in title_lower:
                card_data['type'] = 'puu_ekkukesra'
            elif 'pemantauan' in title_lower and 'pelaksanaan' in title_lower:
                card_data['type'] = 'puspanlakuu'
            else:
                card_data['type'] = 'unknown'
            
            return card_data
            
        except Exception as e:
            print(f"      ❌ Error extracting from card {card_number}: {e}")
            return card_data
    
    def find_and_click_partisipasi(self):
        """Find and click Partisipasi Masyarakat card to get nested content"""
        print("🎯 Looking for Partisipasi Masyarakat card...")
        
        try:
            # Look for card with "Partisipasi" in title or content
            partisipasi_selectors = [
                "//div[contains(@class, 'card') and .//text()[contains(., 'Partisipasi')]]",
                "//div[contains(@class, 'card') and .//h2[contains(text(), 'Partisipasi')]]",
                "//*[contains(text(), 'Partisipasi Masyarakat')]/.."
            ]
            
            partisipasi_element = None
            
            for selector in partisipasi_selectors:
                try:
                    elements = self.driver.find_elements(By.XPATH, selector)
                    for element in elements:
                        if 'partisipasi' in element.text.lower() and 'masyarakat' in element.text.lower():
                            # Look for clickable child (button or link)
                            try:
                                clickable = element.find_element(By.CSS_SELECTOR, "a.btn, button")
                                if clickable and 'sampaikan' in clickable.text.lower():
                                    partisipasi_element = clickable
                                    break
                            except:
                                # Try clicking the card itself
                                partisipasi_element = element
                                break
                    
                    if partisipasi_element:
                        break
                        
                except NoSuchElementException:
                    continue
            
            if not partisipasi_element:
                print("❌ Partisipasi Masyarakat card not found")
                return False
            
            print("🖱️  Clicking Partisipasi Masyarakat card...")
            
            # Click the element
            try:
                self.driver.execute_script("arguments[0].scrollIntoView(true);", partisipasi_element)
                time.sleep(1)
                self.driver.execute_script("arguments[0].click();", partisipasi_element)
                time.sleep(3)  # Wait for new content to load
                
                # Check if we have new content (look for "PUSAT PERANCANGAN")
                if self.wait_for_modal("pusat perancangan"):
                    print("✅ Partisipasi detail modal opened")
                    return True
                else:
                    print("⚠️  Modal opened but checking for different content...")
                    # Sometimes the modal structure is different, let's continue anyway
                    return True
                    
            except Exception as e:
                print(f"❌ Error clicking Partisipasi card: {e}")
                return False
            
        except Exception as e:
            print(f"❌ Error finding Partisipasi card: {e}")
            return False
    
    def extract_partisipasi_detail_cards(self):
        """Extract detailed partisipasi cards (the 3 sub-units)"""
        print("📋 Extracting partisipasi detail cards...")
        
        try:
            # Wait a bit more for content to load
            time.sleep(2)
            
            # Find cards in the new modal/content
            cards = self.find_cards_in_modal()
            
            if not cards:
                print("❌ No partisipasi detail cards found")
                return []
            
            # Extract data from each card
            detail_cards_data = []
            
            for i, card in enumerate(cards, 1):
                try:
                    card_data = self.extract_card_data(card, i, level="partisipasi_detail")
                    if card_data and card_data.get('title'):
                        detail_cards_data.append(card_data)
                        print(f"   ✅ {i}. {card_data['title'][:50]}...")
                    else:
                        print(f"   ❌ {i}. Failed to extract detail card data")
                        
                except Exception as e:
                    print(f"   ❌ {i}. Error extracting detail card: {e}")
                    continue
            
            print(f"✅ Successfully extracted {len(detail_cards_data)} partisipasi detail cards")
            return detail_cards_data
            
        except Exception as e:
            print(f"❌ Error extracting partisipasi detail cards: {e}")
            return []
    
    def scrape_all_aspirasi_data(self):
        """Main scraping function - scrape both levels"""
        
        print("🚀 Starting DPR RI Enhanced Aspirasi scraping...")
        print(f"📅 Scraper started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: Navigate to homepage
            if not self.navigate_to_homepage():
                print("❌ Failed to load homepage")
                return {}
            
            # Step 2: Click Aspirasi to open main modal
            if not self.find_and_click_aspirasi():
                print("❌ Failed to open Aspirasi modal")
                return {}
            
            # Step 3: Extract main aspirasi cards
            print("\n📋 === LEVEL 1: Main Aspirasi Cards ===")
            main_cards = self.extract_main_aspirasi_cards()
            self.aspirasi_data['main_services'] = main_cards
            
            # Step 4: Click Partisipasi to get detailed cards
            print("\n🎯 === LEVEL 2: Partisipasi Detail ===")
            if self.find_and_click_partisipasi():
                detail_cards = self.extract_partisipasi_detail_cards()
                self.aspirasi_data['partisipasi_detail'] = detail_cards
            else:
                print("⚠️  Could not access partisipasi detail, continuing with main data")
            
            # Summary
            total_main = len(self.aspirasi_data['main_services'])
            total_detail = len(self.aspirasi_data['partisipasi_detail'])
            
            print(f"\n🎯 SCRAPING COMPLETED!")
            print(f"📊 Main services: {total_main}")
            print(f"📊 Partisipasi details: {total_detail}")
            print(f"📊 Total items: {total_main + total_detail}")
            
            return self.aspirasi_data
            
        except KeyboardInterrupt:
            print("\n⚠️  Scraping interrupted by user")
            return self.aspirasi_data
        except Exception as e:
            print(f"❌ Unexpected error during scraping: {e}")
            return self.aspirasi_data
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'driver'):
                self.driver.quit()
                print("🧹 WebDriver cleaned up")
        except:
            pass
    
    def save_results(self):
        """Save results to files"""
        
        total_items = len(self.aspirasi_data.get('main_services', [])) + len(self.aspirasi_data.get('partisipasi_detail', []))
        
        if total_items == 0:
            print("❌ No data to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_file = f"dpr_aspirasi_enhanced_{timestamp}.json"
        try:
            save_data = {
                'scraped_at': datetime.now().isoformat(),
                'total_main_services': len(self.aspirasi_data.get('main_services', [])),
                'total_partisipasi_details': len(self.aspirasi_data.get('partisipasi_detail', [])),
                'data': self.aspirasi_data
            }
            
            with open(json_file, 'w', encoding='utf-8') as file:
                json.dump(save_data, file, indent=2, ensure_ascii=False)
            print(f"💾 JSON saved: {json_file}")
        except Exception as e:
            print(f"❌ JSON save error: {e}")
        
        # Save CSV (flattened)
        csv_file = f"dpr_aspirasi_enhanced_{timestamp}.csv"
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=['level', 'type', 'title', 'description', 'action_text', 'action_url', 'image_url'])
                writer.writeheader()
                
                # Write main services
                for service in self.aspirasi_data.get('main_services', []):
                    writer.writerow(service)
                
                # Write partisipasi details
                for detail in self.aspirasi_data.get('partisipasi_detail', []):
                    writer.writerow(detail)
                    
            print(f"💾 CSV saved: {csv_file}")
        except Exception as e:
            print(f"❌ CSV save error: {e}")
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print detailed summary"""
        
        print(f"\n📊 DETAILED SUMMARY:")
        print(f"=" * 50)
        
        main_services = self.aspirasi_data.get('main_services', [])
        partisipasi_details = self.aspirasi_data.get('partisipasi_detail', [])
        
        print(f"📋 Main Aspirasi Services: {len(main_services)}")
        for i, service in enumerate(main_services, 1):
            print(f"   {i}. {service.get('title', 'N/A')} ({service.get('type', 'N/A')})")
        
        print(f"\n🎯 Partisipasi Detail Services: {len(partisipasi_details)}")
        for i, detail in enumerate(partisipasi_details, 1):
            title = detail.get('title', 'N/A')
            url = detail.get('action_url', 'N/A')
            print(f"   {i}. {title[:60]}{'...' if len(title) > 60 else ''}")
            print(f"      🔗 {url}")
        
        # Count services with URLs vs without
        main_with_urls = sum(1 for s in main_services if s.get('action_url'))
        detail_with_urls = sum(1 for d in partisipasi_details if d.get('action_url'))
        
        print(f"\n📊 Services with external URLs:")
        print(f"   Main services: {main_with_urls}/{len(main_services)}")
        print(f"   Detail services: {detail_with_urls}/{len(partisipasi_details)}")

def main():
    print("🏛️  DPR RI ASPIRASI ENHANCED SCRAPER")
    print("Scraping both main aspirasi and nested partisipasi details")
    print("=" * 65)
    
    # Ask user preference
    print("\n🤖 Choose mode:")
    print("1. Visible browser (you can see what's happening) - Recommended")
    print("2. Headless mode (faster, no browser window)")
    
    choice = input("Enter choice (1 or 2, default=1): ").strip()
    headless = choice == "2"
    
    if headless:
        print("⚠️  Headless mode: No browser window will be shown")
    else:
        print("👀 Visible mode: Browser window will open")
    
    scraper = DPRAspirasiEnhancedScraper(headless=headless)
    
    try:
        data = scraper.scrape_all_aspirasi_data()
        
        if data and (data.get('main_services') or data.get('partisipasi_detail')):
            scraper.save_results()
            print("\n✅ SCRAPING COMPLETED SUCCESSFULLY!")
            
            total_items = len(data.get('main_services', [])) + len(data.get('partisipasi_detail', []))
            print(f"🎯 Total aspirasi items scraped: {total_items}")
        else:
            print("\n❌ No data extracted")
            print("💡 Try running in visible mode to see what's happening")
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        scraper.save_results()  # Save partial results
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        
    finally:
        try:
            scraper.cleanup()
        except:
            pass
    
    print("\n🏁 Program finished")

if __name__ == "__main__":
    main()