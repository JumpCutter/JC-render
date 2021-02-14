require './tests/adoctools'

guard 'shell' do
	watch(/^*\.(adoc|conf|dot)$/) {|m|
		AdocTools.convert(m[0])
	}
end

guard 'livereload' do
	AdocTools.copy()
    watch(%r{^.+\.(css|js|html)$})
end
