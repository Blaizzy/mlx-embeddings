# frozen_string_literal: true

# Utility method extensions for String
class String
  def undent
    gsub(/^.{#{(slice(/^ +/) || '').length}}/, "")
  end
end
