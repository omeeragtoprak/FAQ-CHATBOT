import unittest
import json
import numpy as np
from app2 import *
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class FAQTestCase(unittest.TestCase):

    def setUp(self):
        self.faq_list = load_faq("soru_cevap.json")
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.questions = [item['question'] for item in self.faq_list]
        self.question_embeddings = create_embeddings(self.model, self.questions)
    
    def test_faq_responses(self):
        test_questions = [
            "Vodafone hattımın faturası ne kadar oldu?",
            "Hafta sonları para transferi yapma şansım var mı?",
            "Vadesi dolmamış birikim hesabımdan para alabilir miyim?",
            "HGS etiketi nasıl ve hangi yerlerden temin edebilirim?",
            "Banka şubesine uğramadan kredi başvurusu yapmanın yolu nedir?",
            "Garanti BBVA mobil uygulama şifremi hatırlamıyorum, ne yapmalıyım?",
            "Yıllık eğitim maliyetindeki artış, sigorta teminatımı etkiler mi?",
            "Kredi notum düşük, bunu nasıl yükseltebilirim?",
            "Cep telefonuyla kredi başvurusunda bulunabilir miyim?",
            "Cepbank işlemleri iptal oluyo mu?",
            "Garantili Gelecek Hesabı hakkında bilgi verebilir misiniz? Bu hesabın sunduğu avantajlar ve özellikler nelerdir?",
            "Vadeli hesaplardan vade süresi dolmadan para çekilebilir mi?",
            "Vadeli mevduat hesabını kapatmak için hangi adımları izlemeliyim?",
            "Vadeli hesap açılışı ve faiz oranları hakkında bilgi alabilir miyim?",
            "Vadeli hesaba vade süresi içinde para eklenebilir mi?",
            "Biriktiren Hesap'ın detayları ve avantajları nelerdir?",
            "Biriktiren Hesaba ödeme yöntemleri nelerdir?",
            "Biriktiren Hesaptan sınırsız para çekimi yapılabilir mi?",
            "Vadesiz Altın Hesabı'nın avantajları ve özellikleri hakkında bilgi verir misiniz?",
            "Vadesiz Altın Hesabı açma işlemleri nelerdir?"
        ]
        
        correct_answers = [
            "Garanti BBVA İnternet Bankacılığı ve Garanti BBVA Mobil fatura adımlarında Vodafone fatura borcunuzun ne kadar olduğunu öğrenebilirsiniz.",
            "EFT sistemi mevcut işlem saatleri devam etmektedir. FAST sistemi aracılığıyla, 100.000 TL ve altındaki para transferleriniz 7 gün 24 saat yapılabilecektir.",
            "Standart vadeli mevduat hesabından, tüm vadeler için, vade süresince kısmi para çekimi yapılabilmektedir. Bu durumda çekilen tutarın faizi iptal olmaktadır. Buna karşılık hesapta kalan tutar aynı valör, vade ve faiz oranı ile devam eder. Standart vadeli hesabınızda en az hesap açma alt limiti kadar para kalmalıdır. Bu tutarın altına inilmesinin söz konusu olduğu durumda, hesabınızdaki tüm parayı çekmeniz gerekmektedir.",
            "HGS ürünü için ise başvurularınızı;\n\n\nGaranti BBVA İnternet Bankacılığı’nda Başvurular menüsü HGS adımı\n\n\nGaranti BBVA Mobil’de Başvurular menüsü HGS Başvurusu adımı ya da\n\n\nGaranti BBVA Müşteri İletişim Merkezi’nden\n\n\nGaranti BBVA şubelerinden yapabilirsiniz.",
            "Garanti BBVA İnternet Bankacılığı Başvurular menüsü Kredi adımı İhtiyaç Kredisi seçeneğinden kredi başvurusunda bulunabilir , değerlendirme sonucunuzu anında öğrenebilirsiniz. Değerlendirme sonrasında başvurunuzun onaylanması durumunda kredi tutarınız şubeye gitmenize gerek kalmadan hesabınıza aktarılacaktır.",
            "Garanti BBVA İnternet Bankacılığı kullanıcısıysanız, Garanti BBVA İnternet Bankacılığı parolanızla Garanti BBVA Mobil’i kullanabilirsiniz. Eğer Garanti BBVA İnternet Bankacılığı kullanıcısı değilseniz, Parola Al adımından parolanızı anında alıp Garanti BBVA Mobil’i kullanmaya hemen başlayabilirsiniz.\nParolanızı unutursanız, sahip olduğunuz geçerli bir Garanti BBVA kredi kartı veya Paracard numaranız ve bunların şifresi ile Parola Al adımından parolanızı hemen alabilirsiniz.\nGaranti BBVA kartınız yoksa yine Parola Al adımından size uyan yönlendirmeleri takip ederek parolanızı almak için başvuru yapabilirsiniz.",
            "Hayır, sigortanın başlangıcında belirlenen teminat tutarları değiştirilememektedir. Eğitim masraflarının artması durumunda yeni bir Eğitim Sigortası yaptırarak çocuğunuzun teminat dışında kalan eğitim giderlerini de güvence altına alabilirsiniz.",
            "Kredi notu yükseltme yöntemlerinin en önemlisi kredi kartı ve kredi gibi bankalara ait kredili ürün ödemelerini düzenli bir şekilde yapmaktır. Ödemelerinizi zamanında yapamama riski ile karşı karşıya kaldığınız durumlarda kredi notunuzu yükseltmek için kredi ödemelerinizi bütçenize göre yapılandırabilir veya borçlarınızı tek bir kredide birleştirebilirsiniz. Ayrıca kredi kartınızın asgari ödemesinin son ödeme tarihini geçirmeden yapmaya özen göstermeniz kredi notunuzun düşmesinin de önüne geçecektir.\nKredi skorunun yükseltmenin yollarından bir tanesi de Anahtar Kart Programı’na dahil olmaktır. Anahtar Kart Programı, Garanti BBVA kredi kartlarının fırsatlarla dolu dünyasına giriş anahtarıdır. Anahtar Kart Programı'ndan faydalanmak için tek yapmanız gereken, 700 TL'den başlayan teminat tutarını hesabınızda bulundurmanız. Sonrasında hesabınızdaki teminat tutarına bloke koyularak kartınızın başvurusu onaylanacaktır.\nAnahtar Kart Programı'nda seçeceğiniz Garanti BBVA bireysel kredi kartlarından bir tanesi olabilir. Anahtar Kart Programı’na dahil olmanız durumunda ödemeleriniz her 6 ayda bir değerlendirilir. Bankamızda kullandığınız diğer ürünler için de geçerli olmak üzere; ödemelerinizi düzenli yapmanız durumunda kredi skorunuzu yükseltirsiniz. Değerlendirme sonucunda, uygun bulunursa kartınıza bağlı teminat kaldırılacak ve kartınızı herhangi bir teminata bağlı olmadan kullanabileceksiniz.",
            "444 0 335 Garanti BBVA Anında Kredi Hattı ile müşterilerimizin sadece bir telefon uzağındayız!\nÇalışan, esnaf, emekli tüm bireysel müşterilerimiz bireysel kredi ve kredili mevduat hesabı hakkındaki işlemleriniz için bize bu hat üzerinden ulaşabilirsiniz.\nŞubeye gitmeye vaktiniz yok, ancak nakit ihtiyacınız varsa  ihtiyaç kredisi ve kredili mevduat hesabı talepleriniz için Garanti BBVA Anında Kredi Hattı’nı (444 0 335) arayarak kredili ürün başvurusunda bulunabilir, sadece telefon aracılığı ile işlemlerinizi tamamlayabilir ve hesabınıza ihtiyacınız olan tutarın anlık aktarımını sağlayabilirsiniz.\nKredi veya Kredili Mevduat Hesabı başvurunuz için her zaman Garanti BBVA Anında Kredi Hattı’na 444 0 335’i tuşlayarak ulaşabilirsiniz.\nGaranti BBVA olarak müşterilerimizin kredilerini kullandıktan sonra da her zaman yanındayız. Her türlü yenilik ve gelişmede müşterilerimizin ihtiyaçlarına uygun ve dönemsel taleplerini karşılayacak hizmetleri bu hat aracılığıyla müşterilerimize sunuyoruz.\nGaranti BBVA Anında Kredi Hattı’nı aradığınızda, size sunulan tuşlamalarla doğrudan kredi veya kredili mevduat hesabı hakkında ilgili müşteri temsilcisine kolayca bağlanabilirsiniz.\n444 0 335 Garanti BBVA Anında Kredi Hattı’ndan başvurularınızdan önceki, başvurularınız sırasındaki ya da sonrasındaki sorularınız için destek alabilirsiniz.\nGaranti BBVA Anında Kredi Hattı’ndan sadece yeni başvurular için değil mevcuttaki ihtiyaç kredilerinizin yapılandırma işlemlerini de gerçekleştirebilirsiniz.",
            "Evet. CepBank işleminizi gerçekleştirdikten sonra, para gönderdiğiniz kişi gönderilen parayı çekmediği sürece, İnternet Şubesi’nden CepBank menüsüne girerek veya cep telefonunuza \"IPTAL’’ yazdıktan sonra SMS’le size gönderilen referans numaranızı yazıp 3333’e göndererek işlemi iptal edebilirsiniz. Gönderdiğiniz para çekilmiş ise işlemi iptal edemezsiniz.\nAyrıca dilediğiniz zaman para gönderdiğiniz kişiye henüz para çekim işlemini gerçekleştirmediği bilgisini verebilirsiniz.",
            "Garantili Gelecek Hesabı, düzenli birikim yapmanıza olanak sağlayan ve birikimlerinizi enflasyon karşısında koruyan bir hesap türüdür. Garantili Gelecek Hesabı ile, her ay düzenli olarak belirlediğiniz tutarda birikim yapabilir ve bu birikimlerinizin enflasyon karşısında erimemesini sağlayabilirsiniz. Hesabınıza düzenli ödeme talimatı vererek her ay belirlediğiniz tutar kadar birikim yapabilirsiniz. Bu birikimleriniz, enflasyon oranına göre güncellenir ve böylece enflasyon karşısında değer kaybetmez.",
            "Standart vadeli mevduat hesabından, tüm vadeler için, vade süresince kısmi para çekimi yapılabilmektedir. Bu durumda çekilen tutarın faizi iptal olmaktadır. Buna karşılık hesapta kalan tutar aynı valör, vade ve faiz oranı ile devam eder. Standart vadeli hesabınızda en az hesap açma alt limiti kadar para kalmalıdır. Bu tutarın altına inilmesinin söz konusu olduğu durumda, hesabınızdaki tüm parayı çekmeniz gerekmektedir.",
            "Vadeli mevduat hesabınızı kapatmak için, hesabınızın bulunduğu bankanın şubesine giderek ya da internet bankacılığı ve mobil bankacılık gibi dijital kanallar aracılığıyla başvuruda bulunabilirsiniz. Vadeli mevduat hesabınızı kapatmak istediğinizde, hesabınızdaki tüm parayı çekmek isteyip istemediğinizi belirtmeniz gerekmektedir. Ayrıca, vadeli mevduat hesabınızı kapatmadan önce bankanızın vadeli mevduat hesap kapatma politikalarını ve şartlarını öğrenmek önemlidir. Bazı bankalar, vadeli mevduat hesabınızı kapatırken belirli bir sürenin geçmesi gerektiğini veya belirli bir miktar ceza ödemesi yapmanız gerektiğini belirtebilir.",
            "Vadeli hesap açılışı ve faiz oranları bankadan bankaya farklılık gösterebilir. Genellikle bankaların internet bankacılığı veya mobil bankacılık uygulamaları üzerinden vadeli hesap açılışı yapabilirsiniz. Faiz oranları ise belirlenen vadeye göre değişkenlik gösterir. Bankaların sunduğu faiz oranlarını karşılaştırarak size en uygun vadeli hesap seçeneğini bulabilirsiniz.",
            "Vadeli hesaplarda vade süresi içinde genellikle para eklenemez. Ancak, bazı bankalar belirli şartlar altında vadeli hesaplara vade süresi içinde para ekleme imkanı sunabilir. Bu şartlar bankadan bankaya değişebilir, bu nedenle bankanızın vadeli hesap politikalarını öğrenmek önemlidir.",
            "Biriktiren Hesap, düzenli birikim yapmanıza olanak sağlayan ve birikimlerinizi belirli bir faiz oranı ile değerlendiren bir hesap türüdür. Biriktiren Hesap ile, her ay düzenli olarak belirlediğiniz tutarda birikim yapabilir ve bu birikimlerinizi belirli bir faiz oranı ile değerlendirebilirsiniz. Ayrıca, Biriktiren Hesap'ta biriken paranızı istediğiniz zaman çekme imkanına sahipsiniz.",
            "Biriktiren Hesap’a ödeme yapmak için banka şubeleri, internet bankacılığı veya mobil bankacılık gibi çeşitli yöntemler kullanabilirsiniz. Ayrıca, Biriktiren Hesap’a düzenli ödeme talimatı vererek her ay belirlediğiniz tutarda otomatik olarak para yatırılmasını sağlayabilirsiniz.",
            "Biriktiren Hesap'tan sınırsız para çekimi yapılabilir. Ancak, hesabınızda belirli bir bakiye kalmasını sağlamak önemlidir. Biriktiren Hesap'ta biriken paranızı istediğiniz zaman çekme imkanına sahipsiniz.",
            "Vadesiz Altın Hesabı, altın birikimlerinizi güvenli bir şekilde saklamanızı ve değerlendirmenizi sağlayan bir hesap türüdür. Vadesiz Altın Hesabı ile, altın birikimlerinizi banka güvencesi altında saklayabilir ve altınlarınızı dilediğiniz zaman alım satım yaparak değerlendirebilirsiniz. Ayrıca, Vadesiz Altın Hesabı'nda altınlarınızın değeri güncel piyasa fiyatlarına göre belirlenir ve altın fiyatlarındaki değişimlerden yararlanabilirsiniz.",
            "Vadesiz Altın Hesabı açmak için banka şubelerine giderek veya internet bankacılığı ve mobil bankacılık gibi dijital kanallar aracılığıyla başvuruda bulunabilirsiniz. Vadesiz Altın Hesabı açılışı için genellikle kimlik belgeleri ve adres bilgileri gereklidir. Ayrıca, bankaların altın hesap politikalarını öğrenmek ve gerekli belgeleri temin etmek önemlidir."
        ]

        correct_count = 0
        results = []

        for i, question in enumerate(test_questions):
            similar_questions = find_most_similar_questions(question, self.question_embeddings, self.model)
            if similar_questions:
                context = "En benzer sorular:\n" + "\n".join([f"{self.faq_list[idx]['question']} (Benzerlik: {sim:.2f})" for idx, sim in similar_questions])
                faq_items = [(self.faq_list[idx], sim) for idx, sim in similar_questions]
                answer = ask_mistral(question, context, faq_items)
                
                answer_embedding = self.model.encode(answer, convert_to_tensor=True).cpu().numpy()
                correct_answer_embedding = self.model.encode(correct_answers[i], convert_to_tensor=True).cpu().numpy()
                similarity = cosine_similarity([answer_embedding], [correct_answer_embedding])[0][0]
                
                result = f"Soru: {question}\nBeklenen Cevap: {correct_answers[i]}\nModel Cevabı: {answer}\nBenzerlik: {similarity:.2f}\n"
                results.append(result)
                
                if similarity > 0.60: 
                    correct_count += 1
        
        success_rate = correct_count / len(test_questions)
        results.append(f"Başarı Oranı: {success_rate * 100:.2f}%")
        
        with open('testgemma.txt', 'w', encoding='utf-8') as file:
            for result in results:
                file.write(result + '\n')

if __name__ == '__main__':
    unittest.main()