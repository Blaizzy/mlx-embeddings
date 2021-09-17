# frozen_string_literal: true

module Unparser
  # All unparser constants maybe included in other libraries.
  module Constants

    # All unary operators of the ruby language
    UNARY_OPERATORS = %i[
      ! ~ -@ +@
    ].to_set.freeze

    # All binary operators of the ruby language
    BINARY_OPERATORS = %i[
      + - * / & | && || << >> ==
      === != <= < <=> > >= =~ !~ ^
      ** %
    ].to_set.freeze

    # Keywords
    K_DO       = 'do'
    K_DEF      = 'def'
    K_END      = 'end'
    K_BEGIN    = 'begin'
    K_CASE     = 'case'
    K_CLASS    = 'class'
    K_SELF     = 'self'
    K_ENSURE   = 'ensure'
    K_DEFINE   = 'define'
    K_MODULE   = 'module'
    K_RESCUE   = 'rescue'
    K_RETURN   = 'return'
    K_UNDEF    = 'undef'
    K_DEFINED  = 'defined?'
    K_PREEXE   = 'BEGIN'
    K_POSTEXE  = 'END'
    K_SUPER    = 'super'
    K_BREAK    = 'break'
    K_RETRY    = 'retry'
    K_REDO     = 'redo'
    K_NEXT     = 'next'
    K_FALSE    = 'false'
    K_TRUE     = 'true'
    K_IF       = 'if'
    K_AND      = 'and'
    K_ALIAS    = 'alias'
    K_ELSE     = 'else'
    K_ELSIF    = 'elsif'
    K_FOR      = 'for'
    K_NIL      = 'nil'
    K_NOT      = 'not'
    K_IN       = 'in'
    K_OR       = 'or'
    K_UNLESS   = 'unless'
    K_WHEN     = 'when'
    K_WHILE    = 'while'
    K_UNTIL    = 'until'
    K_YIELD    = 'yield'
    K_ENCODING = '__ENCODING__'
    K_EEND     = '__END__'
    K_FILE     = '__FILE__'
    K_THEN     = 'then'

    KEYWORDS = constants.each_with_object([]) do |name, keywords|
      value = const_get(name).freeze
      next unless name.to_s.start_with?('K_')

      keywords << value.to_sym
    end.to_set.freeze

  end # Constants
end # Unparser
