# typed: false
# frozen_string_literal: true

def init
  super

  return if sections.empty?

  sections[:index].place(:internal).before(:private)
end

def internal
  erb(:internal) if object.has_tag?(:api) && object.tag(:api).text == "internal"
end
