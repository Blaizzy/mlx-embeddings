# frozen_string_literal: true

class SoftwareSpec
  undef uses_from_macos

  def uses_from_macos(deps)
    @uses_from_macos_elements ||= []
    @uses_from_macos_elements << deps
  end
end
