# typed: true
# frozen_string_literal: true

# from https://github.com/lsegal/yard/issues/484#issuecomment-442586899
class IgnoreDirectiveDocstringParser < YARD::DocstringParser
  def parse_content(content)
    super(content&.sub(/(\A(typed|.*rubocop)|TODO):.*/m, ""))
  end
end

YARD::Docstring.default_parser = IgnoreDirectiveDocstringParser
