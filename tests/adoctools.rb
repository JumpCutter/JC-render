Bundler.require :default

class AdocTools
	def self.convert(file)
		Asciidoctor.convert_file(file, to_dir: 'dist', mkdirs: true, doctype: 'book',
								safe: 'safe',
                                extensions: [
                                  'asciidoctor-diagram',
                                  'highlightjs-ext'
                                ],
								attributes: {
                               		'docinfodir'=>'public',
                                    'revnumber'=>'v0.0.0',
                               		'revdate'=>'5 February 2021',
                               		'revremark'=>'Fist Release',
                               		'email'=>'support@jumpcutter.com',
                               		'author'=>'JumpCutter OpenSource',
                               		'icons'=>'font',
                               		'experimental'=>'',
                               		'imagesdir'=>'assets',
                               		'homepage'=>'https://jumpcutter.com',
                               		'idprefix'=>'',
                                    'source-highlighter'=>'prettify',
                                    'source-language'=>'ts',
                                    'prettify-theme'=>'https://jmblog.github.io/color-themes-for-google-code-prettify/themes/atelier-savanna-dark.min.css'
                             	})
	end

	def self.run
		Dir.glob('*.adoc') {|file|
			self.convert(file)
		}
	end

    def self.copy
        filename = Dir.getwd + "/dist/README.html"
        if File.exists?(filename)
            Launchy::Browser.run("file://" + filename)
        end
    end

end
