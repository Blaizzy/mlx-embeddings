# typed: strict
# frozen_string_literal: true

module Predicable
  sig { params(attrs: Symbol).void }
  def attr_predicate(*attrs)
    attrs.each do |attr|
      define_method attr do
        instance_variable_get("@#{attr.to_s.sub(/\?$/, "")}") == true
      end
    end
  end
end
